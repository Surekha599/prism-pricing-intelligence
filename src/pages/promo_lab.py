"""Promotion Lab — interactive promotion simulator with optimiser comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import optimizer
from src.ui_components import (ACCENT, BAD, GOOD, MUTED, WARN, fmt_inr, html_table,
                               kpi_row, section, style_fig)

LADDER = [0.0, 0.05, 0.10, 0.15, 0.20]


def render(core):
    prods, elastic, ih, forecast = core["prods"], core["elastic"], core["inv_health"], core["forecast"]
    fc = forecast.groupby("product_id").forecast_units.mean()

    st.markdown('<div class="prism-sub" style="margin-bottom:10px">Design a promotion, compare it against '
                'the no-promotion baseline and a ladder of discount depths, and see which option maximises '
                '<b style="color:#E7ECF5">contribution margin</b> — not just sales.</div>',
                unsafe_allow_html=True)

    with st.container():
        c = st.columns([2.2, 1.4, 1.2, 1.2])
        default_pid = fc.idxmax()  # highest forecast velocity as a sensible default
        pid = c[0].selectbox("Product", sorted(prods.product_id),
                             index=sorted(prods.product_id).index(default_pid),
                             format_func=lambda p: f"{prods.set_index('product_id').product_name[p]}")
        seg = c[1].selectbox("Target segment", ["All Segments", "Deal Seekers", "Premium Loyalists",
                                                "Frequent Budget Buyers", "New Customers", "Window Shoppers"])
        weeks = c[2].slider("Duration (weeks)", 1, 8, 2)
        camp_cost = c[3].slider("Campaign cost (₹L)", 0, 40, 8) * 1e5

    p = prods[prods.product_id == pid].iloc[0]
    e = elastic[elastic.product_id == pid].iloc[0]
    i = ih[ih.product_id == pid].iloc[0]
    base_units = float(fc.get(pid, 3.0))          # weekly baseline

    section("Discount ladder comparison",
            f"{p.product_name} · {weeks}-week window · targeting {seg}")
    rows = []
    for d in LADDER:
        out = optimizer.optimize_product(base_units, p.current_price, p.base_cost,
                                         e.elasticity, p.competitor_price,
                                         campaign_cost=camp_cost if d > 0 else 0, seg=seg,
                                         grid=[d], inv_days=i.inventory_days,
                                         inventory_units=i.closing_inventory, weeks=weeks)
        if out is None:
            continue
        b = out["best"]
        rows.append(dict(Option="No promotion" if d == 0 else f"{d:.0%} discount",
                         Price=f"₹{b['price']:,.0f}", Units=f"{b['units']:,.0f}",
                         Revenue=b["revenue"], GrossMargin=b["revenue"] - b["units"] * p.base_cost * 0,
                         Contribution=b["contribution_margin"],
                         IncrRevenue=b["incr_revenue"], IncrMargin=b["incr_margin"],
                         ROI=(b["incr_margin"] / camp_cost) if (camp_cost and d > 0) else np.nan,
                         _d=d))
    df = pd.DataFrame(rows)
    best_ix = df.IncrMargin.idxmax()
    df["Best"] = ""
    df.loc[best_ix, "Best"] = "★ OPTIMAL"

    html_table(df[["Best", "Option", "Price", "Units", "Revenue", "Contribution",
                   "IncrRevenue", "IncrMargin", "ROI"]],
               num_cols=("Revenue", "Contribution", "IncrRevenue", "IncrMargin", "ROI"),
               formatters=dict(Revenue=fmt_inr, Contribution=fmt_inr, IncrRevenue=fmt_inr,
                               IncrMargin=fmt_inr, ROI=lambda v: f"{v:.2f}x" if pd.notna(v) else "—"))

    c1, c2 = st.columns(2)
    with c1:
        section("Units & contribution by option")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df.Option, y=df.Units, name="Units", marker_color="rgba(124,140,248,.5)",
                             yaxis="y"))
        fig.add_trace(go.Scatter(x=df.Option, y=df.Contribution / 1e5, name="Contribution (₹L)",
                                 mode="lines+markers", line=dict(color=ACCENT, width=2.5),
                                 yaxis="y2"))
        best_label = df.loc[best_ix, "Option"]
        fig.add_vrect(x0=best_label, x1=best_label, line_width=0,
                      fillcolor="rgba(52,211,153,.10)")
        fig.update_layout(yaxis=dict(title="Units"),
                          yaxis2=dict(title="Contribution ₹L", overlaying="y", side="right"),
                          xaxis=dict(tickangle=-12))
        style_fig(fig, 300)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with c2:
        section("Historical promotion performance", "Median uplift & ROI by type (24 months of events)")
        pt = core["promo_types"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=pt.promotion_type, y=pt.median_uplift * 100, name="Median uplift %",
                             marker_color=[GOOD if u > .15 else (WARN if u > .05 else MUTED) for u in pt.median_uplift],
                             text=[f"{u*100:+.0f}%" for u in pt.median_uplift], textposition="outside",
                             textfont=dict(size=10)))
        fig.update_layout(xaxis=dict(tickangle=-18))
        style_fig(fig, 300, yaxis_title="Median sales uplift %")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    section("Caveats the Lab applies automatically")
    st.markdown(f"""
    <div class="prism-card"><ul class="rec-why">
      <li>Higher sales ≠ better performance — options are ranked by <b>incremental contribution margin</b> net of campaign cost.</li>
      <li>Uplift is modelled via this product's elasticity ({e.elasticity:.2f}) × segment boost ({optimizer.SEGMENT_BOOST.get(seg,1):.2f}√) — Deal Seekers amplify response, Premium Loyalists dampen it.</li>
      <li>Margin floor: options pricing below cost × (1 + {optimizer.MIN_MARGIN_FLOOR:.0%}) are excluded.</li>
      <li>Cannibalisation & post-promo dips are visible in historical events ({len(core['promo_eff'])} analysed) — the median event still returns ROI {core['promo_eff'].roi.median():.1f}x.</li>
    </ul></div>""", unsafe_allow_html=True)

    with st.expander("Inspect all analysed promotion events"):
        pe = core["promo_eff"].sort_values("uplift", ascending=False)
        html_table(pe.assign(uplift_pct=(pe.uplift * 100).round(0), roi=pe.roi.round(2))
                   [["promotion_id", "promotion_type", "category", "discount_pct",
                     "duration_days", "uplift_pct", "incremental_units", "roi", "cannibalization"]],
                   num_cols=("uplift_pct", "incremental_units", "roi", "cannibalization"),
                   formatters=dict(incremental_units=lambda v: f"{v:,.0f}",
                                   roi=lambda v: f"{v:.2f}", cannibalization=lambda v: f"{v:+.0%}"
                                   if pd.notna(v) else "—", discount_pct=lambda v: f"{v:.0%}",
                                   uplift_pct=lambda v: f"+{v:.0f}%"),
                   max_rows=25)
