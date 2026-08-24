"""Pricing Intelligence — product-level pricing economics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import optimizer
from src.ui_components import (ACCENT, ACCENT2, BAD, CAT_COLORS, GOOD, MUTED, WARN,
                               fmt_inr, kpi_row, rec_card, section, style_fig)


def render(core):
    prods, tx, panel = core["prods"], core["tx"], core["panel"]
    elastic, ih, forecast, recs = core["elastic"], core["inv_health"], core["forecast"], core["recs"]

    with st.container():
        f = st.columns([1.6, 1.2, 1.2, 1.2, 1.6])
        cat = f[0].selectbox("Category", ["All"] + sorted(prods.category.unique()))
        brand = f[1].selectbox("Brand", ["All"] + sorted(prods.brand.unique()))
        seg = f[2].selectbox("Customer segment", ["All segments"] + sorted(core["seg_summary"].segment))
        region = f[3].selectbox("Region", ["All India", "North", "South", "East", "West", "Central"])
        days = f[4].selectbox("Date range", ["Last 90 days", "Last 6 months", "Full 24 months"])

    subset = prods.copy()
    if cat != "All":
        subset = subset[subset.category == cat]
    if brand != "All":
        subset = subset[subset.brand == brand]

    pid = st.selectbox("Product", subset.sort_values("product_name").product_id,
                       format_func=lambda p: f"{subset.set_index('product_id').product_name[p]}  ·  {subset.set_index('product_id').category[p]}")
    p = prods[prods.product_id == pid].iloc[0]
    e = elastic[elastic.product_id == pid].iloc[0]
    i = ih[ih.product_id == pid].iloc[0]
    fc = forecast[forecast.product_id == pid]
    rec = recs[recs.product_id == pid].iloc[0]

    price, comp, cost = p.current_price, p.competitor_price, p.base_cost
    margin_pct = (price - cost) / price
    base_units = float(fc.forecast_units.mean()) if len(fc) else 3.0

    kpi_row([
        dict(label="Current price", value=f"₹{price:,.0f}"),
        dict(label="Competitor price", value=f"₹{comp:,.0f}", sub=f"gap {price/comp-1:+.1%}"),
        dict(label="Price elasticity", value=f"{e.elasticity:.2f}",
             sub=f"CI [{e.ci_low:.2f}, {e.ci_high:.2f}] · n={int(e.n_weeks)}w"),
        dict(label="Demand forecast", value=f"{base_units:.1f}/wk", sub="8-week average"),
        dict(label="Inventory days", value=f"{i.inventory_days:.0f}", sub=i.status),
        dict(label="Gross margin", value=f"{margin_pct:.1%}", sub=f"₹{price-cost:,.0f}/unit"),
    ])

    left, right = st.columns([1.5, 1])

    with left:
        section("Price vs demand", "Weekly units vs effective price (log-log) with fitted elastic response")
        g = panel[panel.product_id == pid].copy()
        fig = go.Figure(go.Scatter(
            x=g.price, y=g.units, mode="markers",
            marker=dict(color=[BAD if d > .1 else (WARN if d > .03 else ACCENT) for d in g.disc],
                        size=7, opacity=.75),
            text=[f"₹{pr:,.0f} · {u} units<br>{dd:.0%} discount" for pr, u, dd in zip(g.price, g.units, g.disc)],
            hoverinfo="text", showlegend=False))
        xs = np.linspace(g.price.min() * .96, g.price.max() * 1.04, 60)
        E = abs(e.elasticity)
        fit = np.exp(np.log(max(g.units.mean(), .5)) + (-E) * np.log(xs / g.price.mean()))
        fig.add_trace(go.Scatter(x=xs, y=fit, mode="lines", name="fitted demand curve",
                                 line=dict(color=MUTED, width=1.5, dash="dot")))
        fig.update_xaxes(title="Effective price (₹)")
        fig.update_yaxes(title="Units / week")
        style_fig(fig, 330)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.markdown('<span class="note">● teal = list price · amber = promo price · red = deep promo. '
                    'The fitted curve is the constant-elasticity response Q = A·P^(-E).</span>',
                    unsafe_allow_html=True)

        section("8-week demand forecast", "XGBoost · recursive multi-step at current price")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fc.week, y=fc.forecast_units, mode="lines+markers",
                                 line=dict(color=ACCENT, width=2.5),
                                 marker=dict(size=6), name="Forecast"))
        hist = panel[panel.product_id == pid].tail(12)
        fig.add_trace(go.Scatter(x=hist.week, y=hist.units, mode="lines",
                                 line=dict(color=MUTED, width=1.8), name="Actual (recent)"))
        style_fig(fig, 280, yaxis_title="Units / week")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with right:
        section("Recommended action", "PRISM decision engine")
        opt = optimizer.optimize_product(base_units, price, cost, e.elasticity, comp,
                                         i.closing_inventory)
        rec_obj = dict(rec)
        if opt is not None and rec_obj.get("impact"):
            rec_obj["impact"] = dict(rec_obj["impact"])
            rec_obj["impact"].update(discount=opt["best"]["discount"], price=opt["best"]["price"])
        rec_card(rec_obj)

        section("Discount response curve", "Modelled contribution margin by discount depth")
        if opt is not None:
            t = opt["table"]
            best_d = opt["best"]["discount"]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=t.discount * 100, y=t.contribution_margin,
                                 marker_color=[GOOD if abs(d - best_d) < .001 else "rgba(45,212,191,.25)"
                                               for d in t.discount],
                                 text=[f"{c/1e5:.0f}L" for c in t.contribution_margin],
                                 textfont=dict(size=9)))
            fig.update_layout(xaxis_title="Discount %", yaxis_title="Contribution margin (quarter)")
            style_fig(fig, 260)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.markdown(f'<span class="note">Optimum at <b>{best_d:.0%}</b> discount — '
                        f'expected contribution {fmt_inr(opt["best"]["contribution_margin"])} '
                        f'vs {fmt_inr(opt["baseline"]["contribution_margin"])} at list price.</span>',
                        unsafe_allow_html=True)

        seg_el = core["seg_elastic"]
        section("Elasticity by segment", "Who responds to price on this catalog?")
        if len(seg_el):
            fig = go.Figure(go.Bar(
                x=seg_el.elasticity, y=seg_el.segment, orientation="h",
                marker_color=[BAD if v < -1.3 else (WARN if v < -0.8 else ACCENT2) for v in seg_el.elasticity]))
            fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Elasticity (more negative = more sensitive)")
            style_fig(fig, 240)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
