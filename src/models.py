"""PRISM ML layer.

Model 1 — demand forecasting:  XGBoost on weekly product panel
    target   : units sold per product-week
    features : lags, rolling mean, price, discount, competitor gap, seasonality,
               search interest, market demand, inventory days, lifecycle, rating
    split    : temporal (first 80% weeks train / last 20% test)

Model 2 — promotion response:  XGBoost on (product x promo event) outcomes
    target   : uplift ratio (actual units / seasonally-adjusted expected units)
    features : discount depth, promo type, duration, elasticity, category,
               base velocity, campaign cost, festive flag
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_XGB = False

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _make_regressor(n_estimators=350, max_depth=4, lr=0.06, seed=42):
    if HAS_XGB:
        return XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                            learning_rate=lr, subsample=0.9, colsample_bytree=0.85,
                            min_child_weight=3, random_state=seed, n_jobs=2,
                            verbosity=0)
    return GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                     learning_rate=lr, random_state=seed)


def _metrics(y, yhat):
    return dict(mae=float(mean_absolute_error(y, yhat)),
                rmse=float(np.sqrt(mean_squared_error(y, yhat))),
                r2=float(r2_score(y, yhat)))


# ------------------------------------------------------------------ features
def build_weekly_features(panel: pd.DataFrame, prods: pd.DataFrame,
                          signals: pd.DataFrame, inv: pd.DataFrame) -> pd.DataFrame:
    """panel: product_id x week units/price/disc (from transactions)."""
    df = panel.copy()
    df = df.sort_values(["product_id", "week"])
    df["lag_units_1"] = df.groupby("product_id").units.shift(1)
    df["lag_units_2"] = df.groupby("product_id").units.shift(2)
    df["roll4_units"] = df.groupby("product_id").units.transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    df["price_change_pct"] = df.groupby("product_id").price.pct_change().fillna(0)

    sig = signals.copy()
    sig["week"] = pd.to_datetime(sig.date).dt.to_period("W").dt.start_time
    sc = sig.groupby(["category", "week"]).agg(
        competitor_avg_price=("competitor_avg_price", "mean"),
        market_demand=("market_demand_index", "mean"),
        search_interest=("search_interest_index", "mean"),
        seasonality=("seasonality_index", "mean"),
        sentiment=("economic_sentiment_index", "mean")).reset_index()
    df = df.merge(sc, on=["category", "week"], how="left")
    df["competitor_gap_pct"] = df.price / df.competitor_avg_price - 1

    iv = inv.copy()
    iv["week"] = pd.to_datetime(iv.date).dt.to_period("W").dt.start_time
    ivw = iv.groupby(["product_id", "week"]).agg(
        inventory_days=("inventory_days", "mean"),
        closing_inventory=("closing_inventory", "mean"),
        stockout=("stockout_flag", "mean")).reset_index()
    df = df.merge(ivw, on=["product_id", "week"], how="left")
    df["inventory_days"] = (df.groupby("product_id").inventory_days
                            .transform(lambda s: s.ffill()).fillna(45))
    df["closing_inventory"] = (df.groupby("product_id").closing_inventory
                               .transform(lambda s: s.ffill()).fillna(0))
    df["stockout"] = df.stockout.fillna(0)
    for col in ["competitor_avg_price", "market_demand", "search_interest",
                "seasonality", "sentiment", "competitor_gap_pct"]:
        df[col] = df[col].fillna(df.groupby("category")[col].transform("mean")).fillna(100)

    df = df.merge(prods[["product_id", "rating", "product_lifecycle", "base_cost",
                         "current_price", "list_price"]], on="product_id")
    lc_map = {"Launch": 0, "Growth": 1, "Mature": 2, "Decline": 3}
    df["lifecycle_code"] = df.product_lifecycle.map(lc_map)
    cat_codes = {c: i for i, c in enumerate(sorted(df.category.unique()))}
    df["category_code"] = df.category.map(cat_codes)
    df["margin_pct"] = (df.price - df.base_cost) / df.price
    df["price_vs_list"] = df.price / df.list_price - 1
    df["week_num"] = (df.week.dt.year - 2024) * 52 + df.week.dt.isocalendar().week
    df["month"] = df.week.dt.month
    df = df.dropna(subset=["lag_units_1", "lag_units_2"])
    return df, cat_codes


FEATURES = ["lag_units_1", "lag_units_2", "roll4_units", "price_change_pct", "price",
            "disc", "competitor_gap_pct", "market_demand", "search_interest",
            "seasonality", "sentiment", "inventory_days", "closing_inventory",
            "stockout", "rating", "lifecycle_code", "category_code", "margin_pct",
            "price_vs_list", "week_num"]


# ------------------------------------------------------------------ training
def train_demand_model(df: pd.DataFrame):
    weeks = sorted(df.week.unique())
    cut = weeks[int(len(weeks) * 0.8)]
    tr, te = df[df.week < cut], df[df.week >= cut]

    model = _make_regressor()
    model.fit(tr[FEATURES], tr.units)
    base = LinearRegression().fit(tr[FEATURES], tr.units)
    rf = RandomForestRegressor(n_estimators=160, max_depth=9, n_jobs=3, random_state=42)
    rf.fit(tr[FEATURES], tr.units)

    metrics = {
        "xgb": _metrics(te.units, model.predict(te[FEATURES])),
        "linear": _metrics(te.units, base.predict(te[FEATURES])),
        "random_forest": _metrics(te.units, rf.predict(te[FEATURES])),
        "train_weeks": len(weeks), "test_weeks": len(te.week.unique()),
        "n_train": len(tr), "n_test": len(te),
    }
    imp = sorted(zip(FEATURES, model.feature_importances_),
                 key=lambda x: -x[1])
    metrics["feature_importance"] = {k: float(v) for k, v in imp}
    return model, metrics, cut


def forecast_product(model, df: pd.DataFrame, prods: pd.DataFrame, signals: pd.DataFrame,
                     horizon: int = 8, cat_codes: dict | None = None) -> pd.DataFrame:
    """Recursive multi-step forecast at current price / zero discount."""
    cat_codes = cat_codes or {}
    last = df.sort_values("week").groupby("product_id").tail(1).set_index("product_id")
    tail4 = (df.sort_values(["product_id", "week"]).groupby("product_id").tail(4)
             .groupby("product_id").units.mean())
    prods_ix = prods.set_index("product_id")

    sig = signals.copy()
    sig["week"] = pd.to_datetime(sig.date).dt.to_period("W").dt.start_time
    sc = sig.groupby(["category", "week"]).agg(
        competitor_avg_price=("competitor_avg_price", "mean"),
        market_demand=("market_demand_index", "mean"),
        search_interest=("search_interest_index", "mean"),
        seasonality=("seasonality_index", "mean"),
        sentiment=("economic_sentiment_index", "mean")).reset_index()
    future_weeks = sorted(sc.week.unique())[-horizon:]

    rows = []
    state = {}
    for pid, roll in tail4.items():
        state[pid] = dict(u1=float(roll), u2=float(roll), roll=float(roll))
    for wk in future_weeks:
        for pid, st in state.items():
            pr = prods_ix.loc[pid]
            cat = pr.category
            srow = sc[(sc.category == cat) & (sc.week == wk)]
            if not len(srow):
                srow = sc[sc.category == cat].tail(1)
            s = srow.iloc[0]
            price = float(pr.current_price)
            feats = dict(
                lag_units_1=st["u1"], lag_units_2=st["u2"], roll4_units=st["roll"],
                price_change_pct=0.0, price=price, disc=0.0,
                competitor_gap_pct=price / s.competitor_avg_price - 1,
                market_demand=s.market_demand,
                search_interest=s.search_interest,
                seasonality=s.seasonality, sentiment=s.sentiment,
                inventory_days=last.at[pid, "inventory_days"] if pid in last.index else 45,
                closing_inventory=last.at[pid, "closing_inventory"] if pid in last.index else 0,
                stockout=0.0, rating=pr.rating,
                lifecycle_code={"Launch": 0, "Growth": 1, "Mature": 2, "Decline": 3}[pr.product_lifecycle],
                category_code=cat_codes.get(cat, 0),
                margin_pct=(price - pr.base_cost) / price,
                price_vs_list=price / pr.list_price - 1,
                week_num=(pd.Timestamp(wk).year - 2024) * 52 + pd.Timestamp(wk).isocalendar().week)
            x = pd.DataFrame([feats])[FEATURES]
            yhat = max(float(model.predict(x)[0]), 0.0)
            rows.append(dict(product_id=pid, week=pd.Timestamp(wk), forecast_units=yhat,
                             price=price))
            st["u2"], st["u1"] = st["u1"], yhat
            st["roll"] = 0.5 * st["roll"] + 0.5 * yhat
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ promo model
def build_promo_dataset(tx: pd.DataFrame, prods: pd.DataFrame, promos: pd.DataFrame,
                        elasticity) -> pd.DataFrame:
    """(product x promo event) observations with uplift ratio target."""
    if isinstance(elasticity, pd.Series):
        elas = elasticity
    else:
        elas = elasticity.set_index("product_id").elasticity

    t = tx.copy()
    t["transaction_date"] = pd.to_datetime(t.transaction_date)
    d0 = t.transaction_date.min()
    ndays = (t.transaction_date.max() - d0).days + 1
    t["day_idx"] = (t.transaction_date - d0).dt.days

    # daily unit counts per product (fast slicing)
    daily = (t.groupby(["product_id", "day_idx"]).quantity.sum()
             .unstack(fill_value=0).reindex(columns=range(ndays), fill_value=0))
    counts = {pid: row.values for pid, row in daily.iterrows()}

    prods_ix = prods.set_index("product_id")
    cat_of = prods_ix.category.to_dict()
    rating_of = prods_ix.rating.to_dict()

    scoped_products = {}
    for _, pr in promos.iterrows():
        if pr.promotion_type == "No Discount":
            continue
        cats = str(pr.category).split(";")
        scoped_products[pr.promotion_id] = (prods[prods.category.isin(cats)].product_id.tolist()
                                            if "All" not in cats else prods.product_id.tolist())

    rows = []
    for _, pr in promos.iterrows():
        if pr.promotion_type == "No Discount" or pr.promotion_id not in scoped_products:
            continue
        s = (pd.Timestamp(pr.start_date) - d0).days
        e = (pd.Timestamp(pr.end_date) - d0).days
        n_days = max(e - s + 1, 1)
        pre_lo, pre_hi = s - 42, s - 7
        if pre_hi < 0 or s < 0 or e >= ndays:
            continue
        for pid in scoped_products[pr.promotion_id]:
            arr = counts.get(pid)
            if arr is None:
                continue
            base = arr[pre_lo:pre_hi + 1].sum() / 35
            actual = arr[s:e + 1].sum() / n_days
            if base * 35 < 3 or actual * n_days < 2:
                continue
            rows.append(dict(
                product_id=pid, promotion_id=pr.promotion_id,
                uplift_ratio=actual / max(base, 0.05),
                discount_pct=pr.discount_pct, promo_type=pr.promotion_type,
                duration_days=n_days, category=cat_of.get(pid, "-"),
                elasticity=float(elas.get(pid, -1.2)), base_velocity=base,
                campaign_cost=pr.campaign_cost,
                target_deal_seekers=1 if pr.target_segment in ("Deal Seekers", "All") else 0,
                festive=1 if pd.Timestamp(pr.start_date).month in (10, 11, 12, 1) else 0,
                rating=rating_of.get(pid, 4.2)))
    return pd.DataFrame(rows)


PROMO_FEATURES = ["discount_pct", "duration_days", "elasticity", "base_velocity",
                  "campaign_cost", "target_deal_seekers", "festive", "rating"]


def train_promo_model(ds: pd.DataFrame):
    ds = ds.copy()
    ds["promo_code"] = ds.promo_type.astype("category").cat.codes
    ds["cat_code"] = ds.category.astype("category").cat.codes
    feats = PROMO_FEATURES + ["promo_code", "cat_code"]
    n = len(ds)
    cut = int(n * 0.8)
    order = np.random.RandomState(42).permutation(n)
    tr, te = ds.iloc[order[:cut]], ds.iloc[order[cut:]]
    model = _make_regressor(n_estimators=300, max_depth=4, lr=0.06)
    model.fit(tr[feats], np.log1p(tr.uplift_ratio))
    pred = np.expm1(model.predict(te[feats]))
    metrics = _metrics(te.uplift_ratio, pred)
    imp = sorted(zip(feats, model.feature_importances_), key=lambda x: -x[1])
    metrics["feature_importance"] = {k: float(v) for k, v in imp}
    metrics["n_obs"], metrics["features"] = n, feats
    return model, metrics


# ------------------------------------------------------------------ persistence
def save_all(demand_model, demand_metrics, promo_model, promo_metrics, cut_week):
    MODEL_DIR.mkdir(exist_ok=True)
    import joblib
    joblib.dump(dict(demand=demand_model, promo=promo_model, cut=cut_week), MODEL_DIR / "models.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(
        dict(demand=demand_metrics, promo=promo_metrics, has_xgb=HAS_XGB), indent=2))


def load_all():
    import joblib
    blob = joblib.load(MODEL_DIR / "models.joblib")
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    return blob["demand"], blob["promo"], metrics, blob["cut"]
