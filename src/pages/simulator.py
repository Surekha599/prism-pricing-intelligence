"""Scenario Simulator — current state vs simulated state under user assumptions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import optimizer
from src.ui_components import (ACCENT, ACCENT2, BAD, GOOD, MUTED, WARN, fmt_inr,
                               html_table, kpi_row, section, style_fig)


def render(core):
    prods, elastic, ih, forecast = core["prods"], core["elastic"], core["inv_health"], core["forecast"]
    fc = forecast.groupby("product_id").forecast_units.mean()
    seg_el = core["seg_elastic"].set_index("segment").elasticity

    st.markdown('<div class="prism-sub" style="margin-bottom:8px">Change the levers a revenue manager '
                'actually controls — price, discount depth, competitor moves, targeting and stock — and watch '
                'expected demand, revenue, margin and ROI recompute against today\'s baseline.</div>',
                unsafe_allow_html=True)

    fcix = fc.sort_values(ascending=False)
    default_pid = fcix.index[0]
    pid = st.selectbox("Product", sorted(prods.product_id),
                       index=sorted(prods.product_id).index(default_pid),
                       format_func=lambda p: f"{prods.set_index('product_id').product_name[p]}")
    p = prods[prods.product_id == pid].iloc[0]
    e = elastic[elastic.product_id == pid].iloc[0]
    i = ih[ih.product_id == pid].iloc[0]
    base_units_wk = float(fc.get(pid, 3.0))

    c1, c2, c3, c4, c5 = st.columns(5)
    price_delta = c1.slider("Price change %", -20, 20, 0, help="List price move before discounts")
    discount = c2.slider("Discount %", 0, 30, 0)
    comp_delta = c3.slider("Competitor price move %", -15, 15, 0)
    seg = c4.selectbox("Target segment", ["All Segments", "Deal Seekers", "Premium Loyalists",
                                          "Frequent Budget Buyers", "New Customers", "Window Shoppers"])
    weeks = c5.slider("Window (weeks)", 1, 13, 4)

    E = optimizer.clamp_elasticity(e.elasticity)
    boost = optimizer.SEGMENT_BOOST.get(seg, 1.0) ** 0.5

    # ---- current vs simulated ---------------------------------------------
    price0 = p.current_price
    price1 = price0 * (1 + price_delta / 100)
    comp1 = p.competitor_price * (1 + comp_delta / 100)

    def scenario(price, d, comp, use_seg_boost):
        pr = price * (1 - d)
        gap0, gap1 = price0 / p.competitor_price, pr / comp
        comp_adj = np.clip((gap1 / max(gap0, .01)) ** -0.55, .6, 1.6)
        mult = (1 - d) ** (-E * (boost if use_seg_boost else 1.0)) * comp_adj
        units = base_units_wk * weeks * mult
        return dict(units=units, revenue=units * pr, cm=units * (pr - p.base_cost))

    cur = scenario(price0, 0.0, p.competitor_price, False)
    sim = scenario(price1, discount / 100, comp1, True)
    delta_cm = sim["cm"] - cur["cm"]

    verdict = ("Value-accretive" if delta_cm > 0 else "Margin-destructive")
    v_color = GOOD if delta_cm > 0 else BAD

    section("Current state vs simulated state",
            f"{p.product_name} · elasticity {e.elasticity:.2f} · margin {(price0-p.base_cost)/price0:.0%}")

    st.markdown(f"""
    <div class="prism-card" style="display:flex;justify-content:space-between;align-items:center;gap:18px;flex-wrap:wrap">
      <div style="flex:1;min-width:180px">
        <div class="eyebrow">Current state</div>
        <div class="kpi-grid" style="grid-template-columns:repeat(auto-fit,minmax(120px,1fr))">
          <div class="prism-kpi"><div class="label">Price</div><div class="value">₹{price0:,.0f}</div></div>
          <div class="prism-kpi"><div class="label">Units ({weeks}w)</div><div class="value">{cur['units']:,.0f}</div></div>
          <div class="prism-kpi"><div class="label">Revenue</div><div class="value">{fmt_inr(cur['revenue'])}</div></div>
          <div class="prism-kpi"><div class="label">Contribution</div><div class="value">{fmt_inr(cur['cm'])}</div></div>
        </div>
      </div>
      <div style="font-size:22px;color:{MUTED}">→</div>
      <div style="flex:1;min-width:180px;border-left:3px solid {v_color};padding-left:16px">
        <div class="eyebrow" style="color:{v_color}">Simulated · {verdict}</div>
        <div class="kpi-grid" style="grid-template-columns:repeat(auto-fit,minmax(120px,1fr))">
          <div class="prism-kpi"><div class="label">Price</div><div class="value">₹{price1*(1-discount/100):,.0f}</div></div>
          <div class="prism-kpi"><div class="label">Units</div><div class="value">{sim['units']:,.0f}</div></div>
          <div class="prism-kpi"><div class="label">Revenue</div><div class="value" class="num">{fmt_inr(sim['revenue'])}</div></div>
          <div class="prism-kpi"><div class="label">Contribution</div><div class="value" style="color:{v_color}">{fmt_inr(sim['cm'])}</div></div>
        </div>
      </div>
      <div style="text-align:center;min-width:130px">
        <div class="eyebrow">Δ Contribution</div>
        <div style="font-size:26px;font-weight:750;color:{v_color}">{fmt_inr(delta_cm)}</div>
        <div class="note">units {sim['units']/max(cur['units'],.01)-1:+.0%} · revenue {sim['revenue']/max(cur['revenue'],.01)-1:+.0%}</div>
      </div>
    </div>""", unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        section("Impact breakdown")
        fig = go.Figure()
        labels = ["Units", "Revenue (₹L)", "Gross margin (₹L)", "Contribution (₹L)"]
        curv = [cur["units"], cur["revenue"] / 1e5,
                cur["units"] * (price0 - p.base_cost) / 1e5, cur["cm"] / 1e5]
        simv = [sim["units"], sim["revenue"] / 1e5,
                sim["units"] * (price1 * (1 - discount / 100) - p.base_cost) / 1e5, sim["cm"] / 1e5]
        fig.add_trace(go.Bar(x=labels, y=curv, name="Current", marker_color="rgba(139,149,169,.45)"))
        fig.add_trace(go.Bar(x=labels, y=simv, name="Simulated", marker_color=ACCENT))
        fig.update_layout(barmode="group")
        style_fig(fig, 300)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with right:
        section("Sensitivity sweep", "Contribution margin across discount depths at your price & competitor settings")
        sweep_d = np.arange(0, .31, .02)
        sweep_cm = [scenario(price1, d, comp1, True)["cm"] for d in sweep_d]
        fig = go.Figure(go.Scatter(x=sweep_d * 100, y=sweep_cm, mode="lines",
                                   line=dict(color=ACCENT2, width=2.5)))
        fig.add_trace(go.Scatter(x=[discount], y=[sim["cm"]], mode="markers",
                                 marker=dict(size=11, color=v_color, line=dict(width=2, color="#0B0E13")),
                                 name="Your scenario"))
        opt_d = sweep_d[int(np.argmax(sweep_cm))]
        fig.add_vline(x=opt_d * 100, line=dict(color=GOOD, dash="dot"),
                      annotation_text=f"optimum {opt_d:.0%}", annotation_font=dict(size=10, color=GOOD))
        fig.update_xaxes(title="Discount %")
        fig.update_yaxes(title="Contribution (₹)")
        style_fig(fig, 300)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    section("Assumptions & guardrails")
    st.markdown(f"""
    <div class="prism-card"><ul class="rec-why">
      <li>Demand response: constant-elasticity form Q = Q₀·(1−d)<sup>−E</sup> with product elasticity
          <b>{E:.2f}</b> × segment boost <b>{boost:.2f}</b> ({seg}).</li>
      <li>Competitor effect: demand shifts with relative price gap (current gap
          {price1*(1-discount/100)/comp1:+.1%} vs {price0/p.competitor_price:+.1%} today).</li>
      <li>Baseline units from the XGBoost 8-week forecast ({base_units_wk:.1f}/week), scaled to the {weeks}-week window.</li>
      <li>Margin floor: the engine warns when price approaches cost × (1+8%) —
          currently ₹{p.base_cost*1.08:,.0f}.</li>
      <li>Inventory cap: {i.closing_inventory:,.0f} units on hand ({i.inventory_days:.0f} days cover) —
          simulated demand beyond stock would be unfulfillable.</li>
    </ul></div>""", unsafe_allow_html=True)
