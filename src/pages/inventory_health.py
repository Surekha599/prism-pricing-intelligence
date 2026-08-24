"""Inventory Health — command center for stock risk."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import analytics
from src.ui_components import (ACCENT, ACCENT2, BAD, CAT_COLORS, GOOD, MUTED, WARN,
                               fmt_inr, html_table, kpi_row, section, style_fig)


def render(core):
    ih = analytics.inventory_matrix(core["inv_health"])

    total_value = (ih.inventory_value).sum()
    at_risk = ih.value_at_risk.sum()
    dead = ih[ih.status == "Dead / Slow-moving"]
    under = ih[ih.status == "Understock Risk"]

    kpi_row([
        dict(label="Inventory value", value=fmt_inr(total_value), sub=f"{len(ih)} active SKUs"),
        dict(label="Value at risk", value=fmt_inr(at_risk), sub=f"{at_risk/max(total_value,1):.0%} of stock"),
        dict(label="Overstocked SKUs", value=f"{int((ih.inventory_days > 75).sum())}",
             sub=">75 days cover"),
        dict(label="Dead / slow-moving", value=f"{len(dead)}", sub=">120 days cover"),
        dict(label="Stockout risk SKUs", value=f"{len(under)}", sub="<20 days cover"),
        dict(label="Median cover", value=f"{ih.inventory_days.median():.0f} days"),
    ])

    left, right = st.columns([1.5, 1])
    with left:
        section("Margin × inventory matrix", "Bubble = inventory value · quadrant medians shown")
        q_colors = {"High Margin / High Inventory": WARN, "High Margin / Low Inventory": GOOD,
                    "Low Margin / High Inventory": BAD, "Low Margin / Low Inventory": MUTED}
        fig = go.Figure()
        for q, g in ih.groupby("quadrant"):
            fig.add_trace(go.Scatter(
                x=g.inventory_days.clip(0, 200), y=g.gm_pct * 100, mode="markers", name=q,
                marker=dict(size=np.sqrt(g.inventory_value) / 60 + 6, color=q_colors.get(q, ACCENT),
                            opacity=.75, line=dict(width=0)),
                text=[f"{r.product_name}<br>{r.inventory_days:.0f}d · {r.gm_pct:.0%} GM · {fmt_inr(r.inventory_value)}"
                      for r in g.itertuples()], hoverinfo="text"))
        fig.add_hline(y=ih.gm_pct.median() * 100, line=dict(color=MUTED, dash="dot", width=1))
        fig.add_vline(x=75, line=dict(color=MUTED, dash="dot", width=1))
        fig.update_xaxes(title="Inventory days of cover")
        fig.update_yaxes(title="Gross margin %")
        style_fig(fig, 380)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with right:
        section("Playbook by quadrant")
        st.markdown(f"""
        <div class="prism-card"><ul class="rec-why" style="margin:4px 0">
          <li><b style="color:{WARN}">High margin / high inventory ({int((ih.quadrant=='High Margin / High Inventory').sum())})</b> — margin-funded targeted promotions; the optimizer's prime candidates.</li>
          <li><b style="color:{GOOD}">High margin / low inventory ({int((ih.quadrant=='High Margin / Low Inventory').sum())})</b> — hold price, protect availability, accelerate replenishment.</li>
          <li><b style="color:{BAD}">Low margin / high inventory ({int((ih.quadrant=='Low Margin / High Inventory').sum())})</b> — clearance / bundle exits; discounting further destroys contribution.</li>
          <li><b style="color:{MUTED}">Low margin / low inventory ({int((ih.quadrant=='Low Margin / Low Inventory').sum())})</b> — review assortment; candidates for delisting.</li>
        </ul></div>""", unsafe_allow_html=True)

        section("Inventory value by category")
        vc = ih.groupby("category").inventory_value.sum().sort_values()
        fig = go.Figure(go.Bar(x=vc.values, y=vc.index, orientation="h",
                               marker_color="rgba(45,212,191,.55)"))
        fig.update_xaxes(tickvals=[], title="")
        style_fig(fig, 260)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    t1, t2 = st.columns(2)
    with t1:
        section("Dead & slow-moving inventory", ">120 days cover — clearance candidates")
        d = dead.sort_values("inventory_value", ascending=False).head(12)
        if len(d):
            html_table(d[["product_name", "category", "closing_inventory", "inventory_days",
                          "sell_through_90d", "inventory_value"]],
                       num_cols=("closing_inventory", "inventory_days", "sell_through_90d", "inventory_value"),
                       formatters=dict(inventory_value=fmt_inr, sell_through_90d=lambda v: f"{v:.0%}",
                                       inventory_days=lambda v: f"{v:.0f}"))
        else:
            st.markdown('<div class="note">No dead inventory detected in the current cycle.</div>',
                        unsafe_allow_html=True)
    with t2:
        section("Stockout risk", "High velocity with thin cover")
        u = under.sort_values("stockout_probability", ascending=False).head(12)
        if len(u):
            html_table(u[["product_name", "category", "closing_inventory", "inventory_days",
                          "weekly_velocity", "stockout_probability"]],
                       num_cols=("closing_inventory", "inventory_days", "weekly_velocity",
                                 "stockout_probability"),
                       formatters=dict(stockout_probability=lambda v: f"{v:.0%}",
                                       weekly_velocity=lambda v: f"{v:.1f}",
                                       inventory_days=lambda v: f"{v:.0f}"))
        else:
            st.markdown('<div class="note">No SKUs below the 20-day risk threshold.</div>',
                        unsafe_allow_html=True)
