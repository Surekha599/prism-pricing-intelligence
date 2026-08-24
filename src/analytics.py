"""PRISM analytics layer.

Elasticity methodology (defensible in interviews):
  1. product x week panel with category seasonal adjustment
  2. weeks aggregated into distinct price levels (promo depth buckets)
     -> averages out Poisson count noise
  3. within-product OLS: ln(adj_units) ~ ln(price)  => slope = elasticity
  4. category / segment aggregates are precision-weighted (1/se^2)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------------ helpers
def _ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    cov = np.linalg.inv(X.T @ X) * (resid @ resid / dof)
    se = np.sqrt(np.diag(cov))
    return beta, se


# ------------------------------------------------------------------ revenue
def revenue_kpis(tx: pd.DataFrame) -> dict:
    tx = tx.copy()
    tx["transaction_date"] = pd.to_datetime(tx.transaction_date)
    end = tx.transaction_date.max()
    p0, p1 = end - pd.Timedelta(days=89), end
    pp0, pp1 = end - pd.Timedelta(days=179), end - pd.Timedelta(days=90)

    def _k(a, b):
        d = tx[tx.transaction_date.between(a, b)]
        rev, cst = d.revenue.sum(), d.cost.sum()
        return dict(revenue=rev, gm=rev - cst, units=d.quantity.sum(),
                    orders=len(d), aov=rev / max(len(d), 1), disc_cost=(d.revenue * d.discount_pct / (1 - d.discount_pct.clip(0, .95))).sum())

    cur, prev = _k(p0, p1), _k(pp0, pp1)
    out = {"current": cur, "previous": prev, "as_of": end}
    for k in ["revenue", "gm", "units", "aov", "disc_cost"]:
        out[f"{k}_delta"] = (cur[k] / prev[k] - 1) if prev[k] else 0.0
    out["gm_pct"] = cur["gm"] / max(cur["revenue"], 1)
    return out


def weekly_revenue_margin(tx: pd.DataFrame) -> pd.DataFrame:
    t = tx.copy()
    t["week"] = pd.to_datetime(t.transaction_date).dt.to_period("W").dt.start_time
    g = t.groupby("week").agg(revenue=("revenue", "sum"), margin=("gross_margin", "sum"),
                              units=("quantity", "sum"), orders=("transaction_id", "count")).reset_index()
    g["gm_pct"] = g.margin / g.revenue
    return g


def revenue_by_category(tx, prods) -> pd.DataFrame:
    m = tx.merge(prods[["product_id", "category"]], on="product_id")
    g = m.groupby("category").agg(revenue=("revenue", "sum"), margin=("gross_margin", "sum"),
                                  units=("quantity", "sum")).reset_index()
    g["gm_pct"] = g.margin / g.revenue
    return g.sort_values("revenue", ascending=False)


def margin_leakage(tx, prods, promos) -> pd.DataFrame:
    """Discount cost + campaign spend vs incremental margin, by category."""
    m = tx.merge(prods[["product_id", "category"]], on="product_id")
    m["discount_cost"] = m.revenue * m.discount_pct / (1 - m.discount_pct.clip(0, .95))
    leak = m.groupby("category").agg(discount_cost=("discount_cost", "sum"),
                                     gross_margin=("gross_margin", "sum"),
                                     revenue=("revenue", "sum")).reset_index()
    promo_cost = promos.assign(cats=promos.category.str.split(";")).explode("cats")
    promo_cost.loc[promo_cost.cats == "All", "cats"] = np.nan
    pc = promo_cost.dropna(subset=["cats"]).groupby("cats").campaign_cost.sum()
    leak["campaign_cost"] = leak.category.map(pc).fillna(0)
    leak["total_leakage"] = leak.discount_cost + leak.campaign_cost
    leak["leakage_pct_of_rev"] = leak.total_leakage / leak.revenue
    return leak.sort_values("total_leakage", ascending=False)


# ------------------------------------------------------------------ elasticity
def estimate_elasticities(tx: pd.DataFrame, prods: pd.DataFrame, signals: pd.DataFrame):
    t = tx.copy()
    t["transaction_date"] = pd.to_datetime(t.transaction_date)
    t["week"] = t.transaction_date.dt.to_period("W").dt.start_time
    sig = signals.copy()
    sig["week"] = pd.to_datetime(sig.date).dt.to_period("W").dt.start_time
    seas = sig.groupby(["category", "week"]).market_demand_index.mean().rename("seas").reset_index()

    panel = t.groupby(["product_id", "week"]).agg(
        units=("quantity", "sum"), price=("selling_price", "mean"),
        disc=("discount_pct", "mean")).reset_index()
    panel = panel.merge(prods[["product_id", "category"]], on="product_id").merge(seas, on=["category", "week"], how="left")
    panel["seas"] = panel.groupby("category")["seas"].transform(lambda s: s.fillna(s.mean()))
    panel["adj_units"] = panel.units / (panel.seas / 100.0)          # de-seasonalise
    panel["disc_bucket"] = (panel.disc * 100).round(-1) / 100        # ~10pp buckets… refined below

    rows = []
    for pid, g in panel.groupby("product_id"):
        # price levels: effective price rounded to 2.5% grid
        g = g.assign(price_level=(g.price / g.price.max() * 40).round() / 40)
        agg = g.groupby("price_level").agg(adj_units=("adj_units", "mean"),
                                           price=("price", "mean"), n=("week", "count"))
        agg = agg[agg.n >= 3]
        if len(agg) < 3 or agg.price.pct_change().abs().max() < 0.02:
            rows.append((pid, np.nan, np.nan, np.nan, len(g)))
            continue
        X = np.column_stack([np.log(agg.price.values), np.ones(len(agg))])
        beta, se = _ols(np.log(agg.adj_units.values), X)
        rows.append((pid, beta[0], se[0], agg.n.sum(), len(g)))
    est = pd.DataFrame(rows, columns=["product_id", "elasticity", "se", "n_used", "n_weeks"])
    est["t_stat"] = est.elasticity / est.se
    est["ci_low"] = est.elasticity - 1.96 * est.se
    est["ci_high"] = est.elasticity + 1.96 * est.se
    est = est.merge(prods[["product_id", "category"]], on="product_id")

    cat_rows = []
    for cat, g in est.dropna(subset=["elasticity"]).groupby("category"):
        w = 1 / g.se.clip(lower=0.05) ** 2
        e = np.average(g.elasticity, weights=w)
        se_cat = np.sqrt(1 / w.sum())
        cat_rows.append(dict(category=cat, elasticity=e, se=se_cat, n_products=len(g),
                             significant=int((g.t_stat.abs() > 1.96).sum())))
    cat_est = pd.DataFrame(cat_rows)
    return est, cat_est, panel


def estimate_segment_elasticity(tx: pd.DataFrame, cust_seg: pd.DataFrame, prods, signals):
    """Pooled two-way panel: demean ln(units) & ln(price) within product."""
    t = tx.merge(cust_seg[["customer_id", "segment"]], on="customer_id")
    t["transaction_date"] = pd.to_datetime(t.transaction_date)
    t["week"] = t.transaction_date.dt.to_period("W").dt.start_time
    out = []
    for seg, g in t.groupby("segment"):
        p = g.groupby(["product_id", "week"]).agg(units=("quantity", "sum"),
                                                  price=("selling_price", "mean")).reset_index()
        p = p[p.units > 0]
        if len(p) < 400:
            continue
        p["lnq"] = np.log(p.units + .3)
        p["lnp"] = np.log(p.price)
        p["dq"] = p.lnq - p.groupby("product_id").lnq.transform("mean")
        p["dp"] = p.lnp - p.groupby("product_id").lnp.transform("mean")
        keep = p.groupby("product_id").filter(lambda x: x.dp.abs().max() > 0.01
                                              and len(x) >= 12)
        if len(keep) < 200:
            continue
        X = np.column_stack([keep.dp.values, np.ones(len(keep))])
        beta, se = _ols(keep.dq.values, X)
        out.append(dict(segment=seg, elasticity=beta[0], se=se[0],
                        n_obs=len(keep), n_products=keep.product_id.nunique()))
    return pd.DataFrame(out).sort_values("elasticity")


# ------------------------------------------------------------------ promotions
def promotion_effectiveness(tx: pd.DataFrame, prods: pd.DataFrame, promos: pd.DataFrame,
                            signals: pd.DataFrame) -> pd.DataFrame:
    t = tx.copy()
    t["transaction_date"] = pd.to_datetime(t.transaction_date)
    sig = signals.copy()
    sig["date"] = pd.to_datetime(sig.date)
    seas_map = sig.groupby(["category", sig.date.dt.to_period("W").dt.start_time]).market_demand_index.mean()

    rows = []
    for _, pr in promos.iterrows():
        if pr.promotion_type == "No Discount":
            continue
        cats = str(pr.category).split(";")
        scope = prods[prods.category.isin(cats)] if "All" not in cats else prods
        ids = set(scope.product_id)
        s, e = pd.Timestamp(pr.start_date), pd.Timestamp(pr.end_date)

        pre = t[t.transaction_date.between(s - pd.Timedelta(days=42), s - pd.Timedelta(days=7))
                & t.product_id.isin(ids)]
        dur = t[t.transaction_date.between(s, e) & t.product_id.isin(ids)]
        post = t[t.transaction_date.between(e + pd.Timedelta(days=1), e + pd.Timedelta(days=14))
                 & t.product_id.isin(ids)]
        n_days = max((e - s).days + 1, 1)
        pre_daily = len(pre) / 35

        # seasonal ratio during vs pre (category of promo scope)
        cat0 = cats[0] if cats and cats[0] != "All" else "Smartphones"
        wk_d = pd.Timestamp(s).to_period("W").start_time
        wk_p = pd.Timestamp(s - pd.Timedelta(days=21)).to_period("W").start_time
        try:
            seas_ratio = seas_map.get((cat0, wk_d), 100) / max(seas_map.get((cat0, wk_p), 100), 1)
        except Exception:
            seas_ratio = 1.0
        seas_ratio = float(np.clip(seas_ratio, 0.7, 1.5))

        if pre_daily < 0.5 or len(dur) < 5:
            continue
        exp_daily = pre_daily * seas_ratio
        act_daily = len(dur) / n_days
        uplift = act_daily / exp_daily - 1
        inc_units = (act_daily - exp_daily) * n_days
        avg_price = dur.selling_price.mean()
        avg_cost = prods[prods.product_id.isin(ids)].base_cost.mean()
        inc_revenue = inc_units * avg_price
        inc_margin = inc_units * (avg_price - avg_cost) - pr.campaign_cost
        roi = inc_margin / max(pr.campaign_cost, 1)

        # cannibalisation: non-scoped same-category products
        nonscope = set(prods[prods.category.isin(cats) & ~prods.product_id.isin(ids)].product_id) if "All" not in cats else set()
        cann = np.nan
        if nonscope:
            npre = t[t.transaction_date.between(s - pd.Timedelta(days=42), s - pd.Timedelta(days=7))
                     & t.product_id.isin(nonscope)]
            ndur = t[t.transaction_date.between(s, e) & t.product_id.isin(nonscope)]
            if len(npre) > 20:
                cann = (len(ndur) / n_days) / (len(npre) / 35) - 1

        post_dip = (len(post) / 14) / exp_daily - 1 if len(post) else np.nan
        rows.append(dict(promotion_id=pr.promotion_id, promotion_type=pr.promotion_type,
                         category=pr.category if pr.category != "All" else "All categories",
                         discount_pct=pr.discount_pct, start_date=s, end_date=e,
                         duration_days=n_days, target_segment=pr.target_segment,
                         campaign_cost=pr.campaign_cost,
                         base_daily_units=pre_daily, promo_daily_units=act_daily,
                         uplift=uplift, incremental_units=inc_units,
                         incremental_revenue=inc_revenue, incremental_margin=inc_margin,
                         roi=roi, cannibalization=cann, post_promo_dip=post_dip))
    return pd.DataFrame(rows)


def promo_type_summary(promo_eff: pd.DataFrame) -> pd.DataFrame:
    return (promo_eff.groupby("promotion_type")
            .agg(events=("promotion_id", "count"), avg_discount=("discount_pct", "mean"),
                 median_uplift=("uplift", "median"), median_roi=("roi", "median"),
                 avg_cannibalization=("cannibalization", "mean"),
                 total_incremental_margin=("incremental_margin", "sum"))
            .reset_index().sort_values("median_uplift", ascending=False))


# ------------------------------------------------------------------ inventory
def inventory_health(inv: pd.DataFrame, prods: pd.DataFrame) -> pd.DataFrame:
    i = inv.copy()
    i["date"] = pd.to_datetime(i.date)
    last90 = i[i.date >= i.date.max() - pd.Timedelta(days=90)]
    agg = last90.groupby("product_id").agg(
        stockout_rate=("stockout_flag", "mean"),
        avg_inventory=("closing_inventory", "mean"),
        units_sold=("units_sold", "sum")).reset_index()
    cur = i.sort_values("date").groupby("product_id").tail(1)[
        ["product_id", "closing_inventory", "inventory_days"]]
    h = cur.merge(agg, on="product_id").merge(
        prods[["product_id", "category", "product_name", "base_cost", "current_price"]],
        on="product_id")
    h["unit_margin"] = h.current_price - h.base_cost
    h["gm_pct"] = h.unit_margin / h.current_price
    h["inventory_value"] = h.closing_inventory * h.base_cost
    h["weekly_velocity"] = h.units_sold / 90 * 7
    h["sell_through_90d"] = h.units_sold / (h.units_sold + h.avg_inventory).clip(lower=1)
    h["stockout_probability"] = h.stockout_rate.clip(0, 1)
    h["status"] = np.where(h.inventory_days > 120, "Dead / Slow-moving",
                  np.where(h.inventory_days > 75, "Overstock",
                  np.where(h.inventory_days < 20, "Understock Risk", "Healthy")))
    h["value_at_risk"] = np.where(h.inventory_days > 75, h.inventory_value, 0)
    return h


def inventory_matrix(h: pd.DataFrame) -> pd.DataFrame:
    med_m, med_d = h.gm_pct.median(), h.inventory_days.median()
    h = h.copy()
    h["quadrant"] = np.where(
        (h.gm_pct >= med_m) & (h.inventory_days >= med_d), "High Margin / High Inventory",
        np.where((h.gm_pct >= med_m) & (h.inventory_days < med_d), "High Margin / Low Inventory",
        np.where((h.gm_pct < med_m) & (h.inventory_days >= med_d), "Low Margin / High Inventory",
                 "Low Margin / Low Inventory")))
    return h
