"""RFM + behavioural segmentation -> 7 named business segments.

Hybrid approach (deliberate, explainable):
  * rule-based overrides for time-dependent states (New, At-Risk)
  * KMeans (k=5) on standardised RFM + discount sensitivity for the rest
  * clusters mapped to business names by profiling (monetary x frequency x discount)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

SEGMENTS = ["Premium Loyalists", "Deal Seekers", "High-Value Occasional",
            "Frequent Budget Buyers", "Window Shoppers", "At-Risk High Value", "New Customers"]


def build_segments(tx: pd.DataFrame, cust: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = tx.copy()
    t["transaction_date"] = pd.to_datetime(t.transaction_date)
    end = t.transaction_date.max()

    g = t.groupby("customer_id").agg(
        last_purchase=("transaction_date", "max"),
        frequency=("transaction_id", "nunique"),
        monetary=("revenue", "sum"),
        avg_discount=("discount_pct", "mean"),
        units=("quantity", "sum"),
        returns=("return_flag", "sum")).reset_index()
    g["recency_days"] = (end - g.last_purchase).dt.days
    g["aov"] = g.monetary / g.frequency.clip(lower=1)
    g["return_rate"] = g.returns / g.frequency.clip(lower=1)
    g["discount_sensitivity"] = g.avg_discount / g.avg_discount.mean()
    first = t.groupby("customer_id").transaction_date.min().rename("first_purchase")
    g = g.merge(first, on="customer_id")
    g["tenure_days"] = (end - g.first_purchase).dt.days

    feats = g.set_index("customer_id")[["recency_days", "frequency", "monetary", "discount_sensitivity"]].copy()
    for c in ["frequency", "monetary"]:
        feats[c] = np.log1p(feats[c])
    X = StandardScaler().fit_transform(feats)

    km = KMeans(n_clusters=5, n_init=10, random_state=42).fit(X)
    g["cluster"] = km.labels_

    # ---- map clusters to business names by profile ------------------------
    prof = g.groupby("cluster").agg(mon=("monetary", "median"), freq=("frequency", "median"),
                                    disc=("avg_discount", "median"), rec=("recency_days", "median"))
    order = prof.sort_values(["mon"], ascending=False).index.tolist()
    names = {}
    remaining = list(SEGMENTS)
    # highest monetary + high freq -> Premium Loyalists
    names[order[0]] = "Premium Loyalists"
    # most discount-driven remaining -> Deal Seekers
    rest = [c for c in order[1:]]
    dsel = prof.loc[rest, "disc"].idxmax()
    names[dsel] = "Deal Seekers"
    rest.remove(dsel)
    # highest frequency remaining -> Frequent Budget Buyers
    fsel = prof.loc[rest, "freq"].idxmax()
    names[fsel] = "Frequent Budget Buyers"
    rest.remove(fsel)
    # highest monetary of what's left -> High-Value Occasional
    msel = prof.loc[rest, "mon"].idxmax()
    names[msel] = "High-Value Occasional"
    rest.remove(msel)
    names[rest[0]] = "Window Shoppers"

    g["segment"] = g.cluster.map(names)
    # ---- rule overrides for time-dependent states --------------------------
    g.loc[(g.tenure_days <= 90) & (g.frequency <= 3), "segment"] = "New Customers"
    hi_mon = g.monetary.quantile(0.70)
    g.loc[(g.monetary >= hi_mon) & (g.recency_days > 120), "segment"] = "At-Risk High Value"

    seg = g[["customer_id", "recency_days", "frequency", "monetary", "aov",
             "avg_discount", "return_rate", "segment"]].rename(
        columns={"avg_discount": "discount_sensitivity_raw"})
    full = cust.merge(seg, on="customer_id", how="left")
    full["segment"] = full.segment.fillna("Window Shoppers")   # never transacted
    return full, g


def segment_summary(full: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    t = tx.merge(full[["customer_id", "segment"]], on="customer_id")
    t = t.copy()
    t["transaction_date"] = pd.to_datetime(t.transaction_date)
    end = t.transaction_date.max()
    active = t[t.transaction_date >= end - pd.Timedelta(days=90)]

    rows = []
    for seg, gc in full.groupby("segment"):
        st = t[t.customer_id.isin(gc.customer_id)]
        st90 = active[active.customer_id.isin(gc.customer_id)]
        rows.append(dict(
            segment=seg, customers=len(gc),
            share=len(gc) / len(full),
            revenue=st.revenue.sum(),
            revenue_share=st.revenue.sum() / t.revenue.sum(),
            aov=st.revenue.sum() / max(len(st), 1),
            purchase_frequency_90d=len(st90) / max(len(gc), 1),
            avg_discount=st.discount_pct.mean(),
            margin_contribution=st.gross_margin.sum(),
            gm_pct=st.gross_margin.sum() / max(st.revenue.sum(), 1),
            return_rate=st.return_flag.mean(),
            avg_recency=gc.recency_days.mean(),
        ))
    return pd.DataFrame(rows).sort_values("revenue", ascending=False)


def segment_profile(full: pd.DataFrame, tx: pd.DataFrame, prods: pd.DataFrame, segment: str) -> dict:
    gc = full[full.segment == segment]
    st = tx[tx.customer_id.isin(gc.customer_id)]
    m = st.merge(prods[["product_id", "category"]], on="product_id")
    top_cats = m.groupby("category").revenue.sum().sort_values(ascending=False).head(5)
    deal_share = (st.discount_pct > 0.05).mean()
    return dict(
        size=len(gc),
        demographics=dict(
            avg_age=gc.age.mean(), income=gc.income_band.mode().iat[0] if len(gc) else "-",
            top_cities=gc.city.value_counts().head(4).to_dict(),
            loyalty=gc.loyalty_tier.mode().iat[0] if len(gc) else "-",
            regions=gc.region.value_counts().head(3).to_dict()),
        behaviour=dict(
            avg_spend=st.groupby("customer_id").revenue.sum().mean(),
            avg_discount=st.discount_pct.mean(),
            deal_purchase_share=deal_share,
            return_rate=st.return_flag.mean(),
            frequency=st.groupby("customer_id").transaction_id.nunique().mean()),
        top_categories=top_cats.to_dict(),
        avg_recency=gc.recency_days.mean(),
    )


def validate_against_designed(full: pd.DataFrame) -> pd.DataFrame:
    """Crosstab computed vs designed archetypes — shown in About page."""
    ct = pd.crosstab(full.customer_segment, full.segment)
    ct = ct.reindex(index=[s for s in SEGMENTS if s in ct.index],
                    columns=[s for s in SEGMENTS if s in ct.columns], fill_value=0)
    return ct
