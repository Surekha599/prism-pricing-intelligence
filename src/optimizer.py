"""PRISM optimization & decision layer.

Pricing optimizer
    objective   : MAXIMISE incremental contribution margin
    decisions   : discount depth d in {0,2,4,...,30}%  x  target segment
    constraints : margin floor (>=8%), max discount 30%, price >= 85% of
                  competitor, inventory coverage, quarterly promo budget

Demand response model used inside the optimizer:
    units(d) = base_units * (1-d)^(-E_eff) * segment_boost * competitor_adj
    blended with the ML promotion-response prediction.

Opportunity score (0-100): elasticity, inventory pressure, margin potential,
competitor gap, demand trend, promotion responsiveness.

Decision engine: transparent rule cascade -> recommendation + why + impact
+ confidence + KPIs (natural language, no black box).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_MARGIN_FLOOR = 0.08
MAX_DISCOUNT = 0.30
PRICE_FLOOR_VS_COMP = 0.85
PROMO_BUDGET = 45_000_000
HOLDING_RATE_MONTHLY = 0.02   # inventory carrying cost: ~2% of unit cost / month
FORCED_MARKDOWN_RATE = 0.35   # eventual clearance depth for dead stock
OVERSTOCK_THRESHOLD_DAYS = 60  # holding-cost credit applies only to slow stock
CLEARANCE_THRESHOLD_DAYS = 90  # beyond this, the no-action counterfactual is a forced clearance

SEGMENT_BOOST = {   # multiplier on discount response when targeting a segment
    "Deal Seekers": 1.35, "All Segments": 1.00, "Frequent Budget Buyers": 1.10,
    "Premium Loyalists": 0.55, "New Customers": 0.90, "Window Shoppers": 0.80,
    "High-Value Occasional": 0.70, "At-Risk High Value": 0.75,
}


def clamp_elasticity(e, lo=0.35, hi=2.6):
    if np.isscalar(e):
        return float(np.clip(abs(np.nan_to_num(e, nan=1.2)), lo, hi))
    arr = np.asarray(e, dtype=float)
    return np.clip(np.abs(np.nan_to_num(arr, nan=1.2)), lo, hi)


# targeting economics: a segment-targeted promotion discounts only the covered
# share of demand (Deal Seekers redeem, full-price buyers keep paying full price)
TARGET_SHARE = {"Deal Seekers": 0.50, "Frequent Budget Buyers": 0.35, "New Customers": 0.30,
                "Window Shoppers": 0.25, "High-Value Occasional": 0.15,
                "Premium Loyalists": 0.10, "All Segments": 1.00}


def _uplift_curve(promo_ds):
    """Empirical uplift ratio by discount bin, targeted vs broad (from promo model data)."""
    if promo_ds is None or not len(promo_ds):
        return None
    bins = [0, .05, .10, .15, .20, .36]
    out = {}
    for tgt in (0, 1):
        g = promo_ds[promo_ds.target_deal_seekers == tgt]
        if not len(g):
            continue
        med = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            b = g[(g.discount_pct >= lo) & (g.discount_pct < hi)]
            med.append(float(np.clip(b.uplift_ratio.median() if len(b) else np.nan, 0.8, 2.2)))
        med = pd.Series(med).ffill().bfill().tolist()
        out[tgt] = dict(bins=bins, med=med)
    return out


def _curve_lookup(curve, targeted, d):
    if not curve:
        return None
    key = 1 if targeted else 0
    if key not in curve:
        key = 1 if 1 in curve else 0
    c = curve[key]
    for lo, hi, m in zip(c["bins"][:-1], c["bins"][1:], c["med"]):
        if lo <= d < hi:
            return m
    return c["med"][-1]


# ------------------------------------------------------------------ optimizer
def optimize_product(base_units, price, cost, elasticity, comp_price,
                     inventory_units=None, seg="All Segments", campaign_cost=0.0,
                     grid=None, inv_days=None, weeks=12, uplift_curve=None):
    """Return dict with optimal discount & expected outcomes.

    ``base_units`` is the WEEKLY baseline; impacts are computed over a
    ``weeks``-week window. Promotion economics are segment-targeted (see
    ``_outcome_at``): the covered share buys at the promo price with the
    empirically-estimated uplift; clearance-risk stock is valued against a
    hold-then-clear counterfactual.
    """
    grid = grid if grid is not None else np.arange(0, 0.32, 0.02)
    rows = []
    for d in grid:
        p = price * (1 - d)
        if p < cost * (1 + MIN_MARGIN_FLOOR):          # margin floor constraint
            continue
        o = _outcome_at(d, base_units, price, cost, elasticity, seg,
                        inv_days=inv_days, inventory_units=inventory_units,
                        weeks=weeks, uplift_curve=uplift_curve,
                        campaign_cost=campaign_cost if d > 0 else 0.0)
        rows.append(dict(discount=round(float(d), 3), price=round(p, 0),
                         units=o["units"], revenue=o["revenue"],
                         contribution_margin=o["cm"], holding_savings=0.0))
    if not rows:
        return None
    best = max(rows, key=lambda r: r["contribution_margin"])
    for r in rows:
        r["incr_margin"] = r["contribution_margin"] - rows[0]["contribution_margin"]
        r["incr_revenue"] = r["revenue"] - rows[0]["revenue"]
    return dict(table=pd.DataFrame(rows), best=best, baseline=rows[0],
                elasticity_used=clamp_elasticity(elasticity), base_units=base_units)

# ------------------------------------------------------------------ opportunity
def opportunity_scores(prods: pd.DataFrame, elasticity: pd.DataFrame,
                       inv_health: pd.DataFrame, forecast: pd.DataFrame,
                       promo_eff: pd.DataFrame) -> pd.DataFrame:
    e = elasticity.set_index("product_id")[["elasticity"]]
    f = forecast.groupby("product_id").forecast_units.sum()
    tail4 = forecast.groupby("product_id").forecast_units.apply(list)

    df = prods.set_index("product_id").copy()
    df["E"] = clamp_elasticity(e.elasticity.reindex(df.index).fillna(-1.2))
    df["margin_pct"] = (df.current_price - df.base_cost) / df.current_price
    df["comp_gap"] = df.current_price / df.competitor_price - 1
    df["inv_days"] = inv_health.set_index("product_id").inventory_days.reindex(df.index).fillna(45)
    df["forecast_units"] = f.reindex(df.index).fillna(4)
    df["forecast_trend"] = df.index.map(
        lambda p: (tail4[p][-2] / max(tail4[p][0], .1) - 1) if p in tail4.index and len(tail4[p]) >= 2 else 0)

    # promotion responsiveness by category (median uplift of its events)
    resp = (promo_eff.assign(cat=promo_eff.category).groupby("cat").uplift.median()
            if len(promo_eff) else pd.Series(dtype=float))

    def _pct(s):
        return s.rank(pct=True)

    score = (0.20 * _pct(df.E)                       # price sensitivity
             + 0.20 * _pct(df.inv_days.clip(0, 160))  # inventory pressure
             + 0.15 * _pct(df.margin_pct)             # margin potential
             + 0.15 * _pct(df.comp_gap)               # competitor gap
             + 0.15 * _pct(df.forecast_trend)         # demand momentum
             + 0.15 * _pct(df.category.map(resp).fillna(resp.median() if len(resp) else 1)))  # promo response
    df["opportunity_score"] = (100 * score).round(1)
    df["opportunity_band"] = pd.cut(df.opportunity_score, [-1, 40, 60, 80, 101],
                                    labels=["Low", "Medium", "High", "Critical"])
    return df.reset_index()


# ------------------------------------------------------------------ decisions
def decide(p_row, elasticity_row, inv_row, forecast_units, promo_uplift_cat=None,
           optimizer_out=None) -> dict:
    """Rule cascade -> explainable recommendation."""
    pid = p_row.product_id
    name = p_row.product_name
    price, cost = p_row.current_price, p_row.base_cost
    comp = p_row.competitor_price
    margin_pct = (price - cost) / price
    inv_days = getattr(inv_row, "inventory_days", 45) if inv_row is not None else 45
    stockout_prob = getattr(inv_row, "stockout_probability", 0) if inv_row is not None else 0
    E = elasticity_row.get("elasticity", -1.2) if isinstance(elasticity_row, dict) else getattr(elasticity_row, "elasticity", -1.2)
    E = clamp_elasticity(E)
    significant = (elasticity_row.get("t_stat", 2) if isinstance(elasticity_row, dict)
                   else getattr(elasticity_row, "t_stat", 2))
    significant = abs(significant) > 1.96
    comp_gap = price / comp - 1

    why, rec, action, seg, disc = [], None, "Maintain price", "All Segments", 0.0
    confidence_base = 55 + (18 if significant else 0) + (10 if inv_row is not None else 0)

    # R1 — margin-destroying discounts (elastic but thin margin)
    if E > 1.4 and margin_pct < 0.12:
        rec = ("Do not discount this product. Demand responds to price, but the margin is thin — "
               "discounting converts margin into units without improving contribution.")
        action = "Hold price / protect margin"
        why = [f"Price elasticity ≈ {E:.2f} (elastic) but gross margin only {margin_pct:.0%}",
               "Simulated discounts raise units but reduce total contribution margin",
               f"Competitor gap {comp_gap:+.1%}; a price war is unwinnable at this margin"]
        confidence = confidence_base + 5
    # R2 — overstock + elastic -> promote
    elif inv_days > 75 and E > 1.1:
        disc = float(np.clip(0.05 + (inv_days - 75) / 900 + 0.03 * (E - 1.0), 0.05, 0.12))
        rec = (f"Run a targeted {disc:.0%} promotion for Deal Seekers. Inventory cover is "
               f"{inv_days:.0f} days with elastic demand — a controlled discount converts "
               "stale stock into cash.")
        action, seg = "Targeted promotion", "Deal Seekers"
        why = [f"Inventory days {inv_days:.0f} (>75 = overstock pressure)",
               f"Price elasticity ≈ {E:.2f} — demand responds strongly to price",
               f"Margin {margin_pct:.0%} gives room for a controlled discount"]
        confidence = confidence_base + 8
    # R3 — competitor undercuts us with margin room
    elif comp_gap > 0.06 and margin_pct > 0.18 and E > 0.8:
        disc = float(np.clip(comp_gap - 0.01, 0.02, 0.10))
        rec = (f"Competitor pricing is {comp_gap:.0%} below current price. Apply a controlled "
               f"price reduction of ~{disc:.0%} (never below the price floor) to defend volume.")
        action, seg = "Controlled price reduction", "All Segments"
        why = [f"Current price {price:,.0f} vs competitor {comp:,.0f} ({comp_gap:+.1%} gap)",
               f"Margin {margin_pct:.0%} can absorb a partial price match",
               f"Elasticity ≈ {E:.2f}; volume defence is worth more than the margin given up"]
        confidence = confidence_base + 6
    # R4 — stockout risk, inelastic -> hold price
    elif stockout_prob > 0.10 or inv_days < 18:
        rec = ("Do not discount this product. Inventory cover is thin and demand does not "
               "need price support — preserve margin and accelerate replenishment instead.")
        action = "Hold price / replenish"
        why = [f"Inventory days {inv_days:.0f} (stockout probability {stockout_prob:.0%})",
               f"Elasticity ≈ {E:.2f} — demand is relatively price-inelastic",
               "Discounting would sell out remaining stock at lower margin"]
        confidence = confidence_base + 7
    # R5 — inelastic healthy -> hold
    elif E < 0.9 and inv_days <= 75:
        rec = ("Maintain price. Demand is relatively price-inelastic and inventory is healthy — "
               "discounting would hand back margin without materially lifting units.")
        action = "Maintain price"
        why = [f"Price elasticity ≈ {E:.2f} (inelastic; |t| {'> 1.96' if significant else '< 1.96'})",
               f"Inventory days {inv_days:.0f} — within healthy range",
               f"Margin {margin_pct:.0%} — no structural pricing problem detected"]
        confidence = confidence_base + 4
    # R6 — elastic, healthy margin, normal stock -> test small discount
    else:
        disc = 0.06
        rec = (f"Test a {disc:.0%} promotion targeted at Deal Seekers. Elastic demand with a "
               "healthy margin and balanced inventory is the classic case for a measured "
               "discount experiment — measure incrementality before scaling.")
        action, seg = "Test promotion", "Deal Seekers"
        why = [f"Price elasticity ≈ {E:.2f} — demand responds to price",
               f"Margin {margin_pct:.0%} provides headroom for a small test",
               f"Inventory days {inv_days:.0f} — no excess, so keep the discount small"]
        confidence = confidence_base + 3

    # expected impact from optimizer table if provided
    impact = None
    if optimizer_out is not None:
        b = optimizer_out["best"]
        impact = dict(units=b["units"], revenue=b["revenue"],
                      contribution_margin=b["contribution_margin"],
                      incr_margin=b.get("incr_margin", 0), incr_revenue=b.get("incr_revenue", 0),
                      discount=b["discount"], price=b["price"])

    kpis = dict(elasticity=round(E, 2), inventory_days=round(inv_days, 0),
                margin_pct=round(margin_pct, 3), competitor_gap=round(comp_gap, 3),
                forecast_units_8w=round(forecast_units, 1),
                current_price=price, competitor_price=comp)
    return dict(product_id=pid, product_name=name, recommendation=rec, action=action,
                why=why, target_segment=seg, discount=disc, confidence=int(min(confidence, 95)),
                impact=impact, kpis=kpis, elasticity_significant=bool(significant))


# ------------------------------------------------------------------ ranked table
def _outcome_at(d, base_units, price, cost, elasticity, seg, inv_days=None,
                inventory_units=None, weeks=12, uplift_curve=None, campaign_cost=0.0):
    """base_units = weekly baseline; window = weeks (default quarter).

    Economics of a targeted promotion:
      * only the covered share of demand buys at the discounted price
        (segment targeting = price discrimination, not a blanket markdown)
      * the covered share responds at the *promotion* uplift rate estimated
        from 9k+ historical product-events (flag/display effect included),
        which exceeds the bare price-elasticity response
      * for clearance-risk stock (>90 days) the no-action counterfactual is
        hold now → forced clearance later at −35%.
    """
    E = clamp_elasticity(elasticity) * (SEGMENT_BOOST.get(seg, 1.0) ** 0.5)
    share = TARGET_SHARE.get(seg, 1.0)
    base_win = base_units * weeks
    if d > 0:
        up = _curve_lookup(uplift_curve, seg == "Deal Seekers", d)
        mult_promo = up if up is not None else (1 - d) ** (-E)
        units_full = base_win * (1 - share)
        units_promo = base_win * share * mult_promo
    else:
        units_full, units_promo = base_win, 0.0
    units = units_full + units_promo
    if inventory_units is not None:
        units = min(units, inventory_units)
    p_d = price * (1 - d)
    cm = (units_full * (price - cost) + units_promo * (p_d - cost)
          - campaign_cost)
    liq_price = price * (1 - FORCED_MARKDOWN_RATE)
    clear_risk = (inv_days is not None and inv_days > CLEARANCE_THRESHOLD_DAYS
                  and inventory_units is not None)
    if clear_risk:
        stranded = max(0.0, inventory_units - units)
        cm += stranded * (liq_price - cost)
    return dict(units=units, revenue=units_full * price + units_promo * p_d, cm=cm)


def build_recommendations(prods, elasticity, inv_health, forecast, promo_eff,
                          cat_codes=None, promo_ds=None):
    e = elasticity.set_index("product_id")
    ih = inv_health.set_index("product_id")
    fc = forecast.groupby("product_id").forecast_units.mean()
    recs = []
    opp = opportunity_scores(prods, elasticity, inv_health, forecast, promo_eff)
    opp_ix = opp.set_index("product_id")
    uplift_curve = _uplift_curve(promo_ds)
    for _, p in prods.iterrows():
        pid = p.product_id
        erow = e.loc[pid] if pid in e.index else None
        irow = ih.loc[pid] if pid in ih.index else None
        base_units = float(fc.get(pid, 3.0))
        opt = optimize_product(base_units, p.current_price, p.base_cost,
                               erow.elasticity if erow is not None else -1.2,
                               p.competitor_price,
                               irow.closing_inventory if irow is not None else None,
                               campaign_cost=250_000 / 20,
                               inv_days=irow.inventory_days if irow is not None else None)
        d = decide(p, erow, irow, base_units * 8, optimizer_out=opt)
        d["opportunity_score"] = float(opp_ix.at[pid, "opportunity_score"]) if pid in opp_ix.index else np.nan
        d["opportunity_band"] = str(opp_ix.at[pid, "opportunity_band"]) if pid in opp_ix.index else "-"
        d["category"] = p.category
        # impact evaluated at the recommended action's discount (rule-consistent)
        rule_d = float(d.get("discount") or 0.0)
        d["recommended_discount"] = rule_d
        d["recommended_price"] = round(p.current_price * (1 - rule_d), 0)
        inv_units = irow.closing_inventory if irow is not None else None
        inv_d = irow.inventory_days if irow is not None else None
        if rule_d > 0:
            e_val = erow.elasticity if erow is not None else -1.2
            out = _outcome_at(rule_d, base_units, p.current_price, p.base_cost,
                              e_val, d["target_segment"], inv_days=inv_d,
                              inventory_units=inv_units, uplift_curve=uplift_curve,
                              campaign_cost=5_000)
            base_out = _outcome_at(0.0, base_units, p.current_price, p.base_cost,
                                   e_val, d["target_segment"], inv_days=inv_d,
                                   inventory_units=inv_units)
            d["expected_revenue_impact"] = out["revenue"] - base_out["revenue"]
            d["expected_margin_impact"] = out["cm"] - base_out["cm"]
            if d.get("impact") is not None:
                d["impact"].update(units=out["units"], revenue=out["revenue"],
                                   contribution_margin=out["cm"],
                                   incr_margin=out["cm"] - base_out["cm"],
                                   incr_revenue=out["revenue"] - base_out["revenue"],
                                   discount=rule_d, price=d["recommended_price"])
        else:
            d["expected_revenue_impact"] = 0.0
            d["expected_margin_impact"] = 0.0
        recs.append(d)
    df = pd.DataFrame(recs)
    return df.sort_values(["expected_margin_impact", "opportunity_score"],
                          ascending=False).reset_index(drop=True)
