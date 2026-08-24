"""Executive Overview — Revenue Command Center."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import analytics
from src.ui_components import (ACCENT, ACCENT2, BAD, CAT_COLORS, GOOD, MUTED, WARN,
                               fmt_inr, kpi_row, section, style_fig)


def render(core):
    tx, prods, promos = core["tx"], core["prods"], core["promos"]
    k = analytics.revenue_kpis(tx)
    wk = analytics.weekly_revenue_margin(tx)
    cat_rev = analytics.revenue_by_category(tx, prods)
    leak = analytics.margin_leakage(tx, prods, promos)
    ih = core["inv_health"]
    recs = core["recs"]

    # ---- KPI cards --------------------------------------------------------
    promo_roi = core["promo_eff"].roi.median()
    inv_risk = ih.value_at_risk.sum()
    kpi_row([
        dict(label="Revenue (90d)", value=fmt_inr(k["current"]["revenue"]), delta=k["revenue_delta"]),
        dict(label="Gross Margin (90d)", value=fmt_inr(k["current"]["gm"]), delta=k["gm_delta"]),
        dict(label="Avg Order Value", value=f"₹{k['current']['aov']:,.0f}", delta=k["aov_delta"]),
        dict(label="Units Sold", value=f"{k['current']['units']:,.0f}", delta=k["units_delta"]),
        dict(label="Promotion ROI (median)", value=f"{promo_roi:.1f}x",
             sub="half of events destroy margin" if promo_roi < 0 else ""),
        dict(label="Inventory at Risk", value=fmt_inr(inv_risk), sub=f"{int((ih.inventory_days>75).sum())} SKUs overstocked"),
    ])

    left, right = st.columns([2.35, 1])

    with left:
        # ---- revenue + margin trend --------------------------------------
        section("Revenue & margin trend", "Weekly, 24 months · shaded bands = festive peaks")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wk.week, y=wk.revenue, name="Revenue", fill="tozeroy",
                                 mode="lines", line=dict(color=ACCENT, width=2),
                                 fillcolor="rgba(45,212,191,.08)"))
        fig.add_trace(go.Scatter(x=wk.week, y=wk.margin, name="Gross margin",
                                 mode="lines", line=dict(color=ACCENT2, width=2)))
        for yr, festive in [(2024, ("2024-10-05", "2024-11-03")),
                            (2025, ("2025-10-05", "2025-10-21")),
                            (2026, None)]:
            if festive:
                fig.add_vrect(x0=festive[0], x1=festive[1], fillcolor="rgba(124,140,248,.06)",
                              line_width=0, annotation_text=f"festive {yr}",
                              annotation_font=dict(size=10, color=MUTED))
        style_fig(fig, 300, yaxis_title="INR / week")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        c1, c2 = st.columns(2)
        with c1:
            section("Revenue by category")
            fig = go.Figure(go.Bar(
                x=cat_rev.revenue, y=cat_rev.category, orientation="h",
                marker_color=[CAT_COLORS[i % 9] for i in range(len(cat_rev))],
                text=[fmt_inr(v) for v in cat_rev.revenue], textposition="outside",
                textfont=dict(size=10)))
            fig.update_layout(yaxis=dict(autorange="reversed"))
            style_fig(fig, 290)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        with c2:
            section("Margin leakage from promotions", "Discount cost + campaign spend vs revenue")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=leak.category, y=leak.discount_cost, name="Discount cost",
                                 marker_color=BAD, marker_opacity=.85))
            fig.add_trace(go.Bar(x=leak.category, y=leak.campaign_cost, name="Campaign spend",
                                 marker_color=WARN, marker_opacity=.85))
            fig.update_layout(barmode="stack")
            fig.update_xaxes(tickangle=-32, tickfont=dict(size=10))
            style_fig(fig, 290)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        # ---- inventory risk matrix ----------------------------------------
        section("Inventory risk matrix", "Gross margin vs inventory cover · bubble = value at risk")
        fig = go.Figure(go.Scatter(
            x=ih.inventory_days.clip(0, 200), y=ih.gm_pct * 100, mode="markers",
            marker=dict(size=np.sqrt(ih.inventory_value) / 40 + 5,
                        color=ih.inventory_value, colorscale="Teal", showscale=True,
                        colorbar=dict(title="Inv value", thickness=8, len=.7, x=1.0),
                        line=dict(width=0)),
            text=[f"{r.product_name}<br>{r.inventory_days:.0f} days · {fmt_inr(r.inventory_value)}"
                  for r in ih.itertuples()], hoverinfo="text"))
        fig.add_hline(y=ih.gm_pct.median() * 100, line=dict(color=MUTED, width=1, dash="dot"))
        fig.add_vline(x=75, line=dict(color=MUTED, width=1, dash="dot"))
        style_fig(fig, 290, xaxis_title="Inventory days", yaxis_title="Gross margin %")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with right:
        section("AI / analytics recommendations",
                "Generated by the PRISM decision engine")
        top = recs.sort_values("opportunity_score", ascending=False)
        shown = 0
        for _, r in top.iterrows():
            if shown >= 6:
                break
            if r.action in ("Maintain price",) and shown >= 3:
                continue
            st.markdown(f"""
            <div class="rec-card" style="border-left-color:{BAD if r.opportunity_band=='Critical' else WARN}">
              <div class="rec-title" style="font-size:12.5px">{r.product_name}</div>
              <div class="rec-body">{r.recommendation[:180]}…</div>
              <div class="rec-meta">
                <span class="badge b-accent" style="margin-top:6px">{r.action}</span>
                {ui_band(r.opportunity_band)}
              </div>
            </div>""", unsafe_allow_html=True)
            shown += 1

        section("Portfolio signals")
        overstock = int((ih.inventory_days > 75).sum())
        overdisc = int(core["elastic"].merge(
            core["prods"][["product_id", "current_price", "base_cost"]], on="product_id")
            .query("current_price / base_cost < 1.12").shape[0])
        comp_gap = core["prods"].assign(gap=core["prods"].current_price / core["prods"].competitor_price - 1)
        gap_cats = comp_gap.groupby("category").gap.median().sort_values()
        st.markdown(f"""
        <div class="prism-card">
          <ul class="rec-why">
            <li><b style="color:{WARN}">{overstock} products</b> have high inventory pressure (>75 days cover)</li>
            <li><b style="color:{BAD}">{overdisc} products</b> are being over-discounted (GM &lt; 12%)</li>
            <li>Premium customers show <b>low discount sensitivity</b> (E ≈ {core['seg_elastic'].set_index('segment').elasticity.get('Premium Loyalists', -0.6):.2f})</li>
            <li><b>{(gap_cats > 0.04).sum()} categories</b> price above competitors by >4%</li>
            <li>Median promo ROI is <b style="color:{BAD}">{core['promo_eff'].roi.median():.1f}x</b> — promotions need targeting, not scaling</li>
          </ul>
        </div>""", unsafe_allow_html=True)


def ui_band(band):
    cls = {"Critical": "b-critical", "High": "b-high", "Medium": "b-medium", "Low": "b-low"}.get(str(band), "b-low")
    return f'<span class="badge {cls}" style="margin-top:6px">{band}</span>'
