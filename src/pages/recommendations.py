"""Product Recommendations — ranked, explainable pricing actions."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui_components import (ACCENT, BAD, GOOD, MUTED, WARN, band_badge, fmt_inr,
                               html_table, rec_card, section, style_fig)

SORTS = {"Margin opportunity": "expected_margin_impact",
         "Revenue opportunity": "expected_revenue_impact",
         "Inventory risk": "opportunity_score",
         "Confidence": "confidence"}


def render(core):
    recs = core["recs"].copy()

    st.markdown('<div class="prism-sub" style="margin-bottom:8px">Every product was passed through the '
                'constrained optimizer (objective: maximise incremental contribution margin) and a transparent '
                'rule cascade. Select any row for the full rationale.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.2, 1.2, 2.6])
    sort_by = c1.radio("Rank by", list(SORTS), horizontal=True, label_visibility="collapsed")
    band = c2.selectbox("Opportunity band", ["All", "Critical", "High", "Medium", "Low"])
    search = c3.text_input("Search product", "", placeholder="e.g. NovaSound, Buds, Titan…")

    df = recs.copy()
    if band != "All":
        df = df[df.opportunity_band == band]
    if search:
        df = df[df.product_name.str.contains(search, case=False)]
    df = df.sort_values(SORTS[sort_by], ascending=False).reset_index(drop=True)
    df["priority"] = df.index + 1

    section(f"Ranked actions — {len(df)} products",
            "Actions are optimizer-validated; 'hold' recommendations are as deliberate as promotions")

    table = df.head(20).copy()
    rows_html = []
    for _, r in table.iterrows():
        imp_cls = "up" if r.expected_margin_impact >= 0 else "down"
        rows_html.append(f"""
        <tr>
          <td>{r.priority}</td>
          <td><b>{r.product_name}</b><br><span class="note">{r.category}</span></td>
          <td><span class="badge b-accent">{r.action}</span></td>
          <td class="num">₹{r.recommended_price:,.0f}</td>
          <td class="num">{r.recommended_discount:.0%}</td>
          <td>{r.target_segment}</td>
          <td class="num {'up' if r.expected_revenue_impact>=0 else 'down'}">{fmt_inr(r.expected_revenue_impact)}</td>
          <td class="num {imp_cls}">{fmt_inr(r.expected_margin_impact)}</td>
          <td>{band_badge(r.opportunity_band)}<br><span class="note">{r.confidence}% conf</span></td>
        </tr>""")
    st.markdown(f"""
    <div class="prism-card" style="padding:6px 12px">
    <table class="prism-table">
      <thead><tr><th>#</th><th>Product</th><th>Action</th><th class="num">Rec. price</th>
      <th class="num">Disc.</th><th>Target</th><th class="num">Rev. impact</th>
      <th class="num">Margin impact</th><th>Opportunity</th></tr></thead>
      <tbody>{''.join(rows_html)}</tbody>
    </table></div>""", unsafe_allow_html=True)

    # action mix
    c1, c2 = st.columns([1, 1.4])
    with c1:
        section("Action mix across the portfolio")
        mix = recs.action.value_counts()
        colors = {"Test promotion": ACCENT, "Maintain price": MUTED,
                  "Hold price / replenish": "#60A5FA", "Controlled price reduction": WARN,
                  "Targeted promotion": GOOD, "Hold price / protect margin": BAD}
        fig = go.Figure(go.Bar(x=mix.values, y=mix.index, orientation="h",
                               marker_color=[colors.get(a, ACCENT) for a in mix.index]))
        fig.update_layout(yaxis=dict(autorange="reversed"))
        style_fig(fig, 260)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with c2:
        section("Opportunity score distribution", "0–100 composite of 6 drivers")
        fig = go.Figure(go.Histogram(
            x=recs.opportunity_score, nbinsx=24,
            marker_color="rgba(124,140,248,.55)"))
        for x0, x1, name, c in [(0, 40, "Low", "rgba(139,149,169,.5)"), (40, 60, "Medium", "rgba(124,140,248,.5)"),
                                (60, 80, "High", "rgba(245,184,76,.5)"), (80, 100, "Critical", "rgba(248,113,113,.5)")]:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=c, line_width=0, annotation_text=name,
                          annotation_font=dict(size=9, color=MUTED))
        style_fig(fig, 260, xaxis_title="Pricing opportunity score")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.markdown("---")
    section("Full rationale explorer")
    sel = st.selectbox("Product", df.head(40).product_name.tolist()
                       if len(df) else recs.head(40).product_name.tolist())
    r = (recs[recs.product_name == sel].iloc[0])
    rec_card(dict(r))
