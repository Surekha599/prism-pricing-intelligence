"""Customer Segments — RFM + behavioural segmentation with drilldown."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import segmentation
from src.ui_components import (ACCENT, ACCENT2, BAD, CAT_COLORS, GOOD, MUTED, WARN,
                               fmt_inr, html_table, kpi_row, section, style_fig)


def render(core):
    seg_sum = core["seg_summary"]
    seg_el = core["seg_elastic"]
    seg_full = core["seg_full"]

    c1, c2 = st.columns([1.4, 1])
    with c1:
        section("Segment landscape", "Recency vs monetary value · bubble = customer count")
        agg = seg_full.groupby("segment").agg(
            recency=("recency_days", "mean"), monetary=("monetary", "median"),
            n=("customer_id", "size"), aov=("aov", "median")).reset_index()
        fig = go.Figure(go.Scatter(
            x=agg.recency, y=agg.monetary, mode="markers+text",
            marker=dict(size=np.sqrt(agg.n) / 2.2 + 10,
                        color=[ACCENT, BAD, ACCENT2, GOOD, MUTED, WARN, "#60A5FA"][:len(agg)],
                        opacity=.85, line=dict(color="#0B0E13", width=1.5)),
            text=agg.segment, textposition="top center", textfont=dict(size=10.5, color="#E7ECF5"),
            hovertext=[f"{r.segment}<br>{r.n:,} customers<br>median spend {fmt_inr(r.monetary)}"
                       for _, r in agg.iterrows()], hoverinfo="text"))
        fig.update_xaxes(title="Avg recency (days) →  stale")
        fig.update_yaxes(title="Median lifetime spend (₹)")
        style_fig(fig, 330)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with c2:
        section("Revenue contribution by segment")
        fig = go.Figure(go.Bar(
            x=seg_sum.revenue_share * 100, y=seg_sum.segment, orientation="h",
            marker_color=[ACCENT if s == selected_hint(seg_sum) else "rgba(124,140,248,.55)"
                          for s in seg_sum.segment]))
        fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="% of total revenue")
        style_fig(fig, 330)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    section("Segment scorecard")
    html_table(seg_sum.assign(
        share=lambda d: (d.share * 100).round(1), revenue_share=lambda d: (d.revenue_share * 100).round(1),
        gm_pct=lambda d: (d.gm_pct * 100).round(1), avg_discount=lambda d: (d.avg_discount * 100).round(1),
        return_rate=lambda d: (d.return_rate * 100).round(1),
        revenue=lambda d: d.revenue, margin=lambda d: d.margin_contribution,
    )[["segment", "customers", "share", "revenue_share", "revenue", "margin", "aov",
       "purchase_frequency_90d", "avg_discount", "gm_pct", "return_rate"]],
        num_cols=("revenue", "margin", "aov"),
        formatters=dict(revenue=fmt_inr, margin=fmt_inr, aov=lambda v: f"₹{v:,.0f}",
                        share=lambda v: f"{v}%", revenue_share=lambda v: f"{v}%",
                        gm_pct=lambda v: f"{v}%", avg_discount=lambda v: f"{v}%",
                        return_rate=lambda v: f"{v}%", purchase_frequency_90d=lambda v: f"{v:.2f}"))

    st.markdown("---")
    sel = st.selectbox("Drill into a segment", seg_sum.segment,
                       index=list(seg_sum.segment).index("Deal Seekers")
                       if "Deal Seekers" in list(seg_sum.segment) else 0)
    _profile(core, sel)


def selected_hint(seg_sum):
    return seg_sum.sort_values("revenue").segment.iat[-1]


def _profile(core, seg):
    left, right = st.columns([1, 1.2])
    with left:
        section(f"Who they are — {seg}")
        full = core["seg_full"]
        gc = full[full.segment == seg]
        cities = gc.city.value_counts(normalize=True).head(4)
        st.markdown(f"""
        <div class="prism-card">
          <div class="kpi-grid">
            <div class="prism-kpi"><div class="label">Customers</div><div class="value">{len(gc):,}</div></div>
            <div class="prism-kpi"><div class="label">Avg age</div><div class="value">{gc.age.mean():.0f}</div></div>
            <div class="prism-kpi"><div class="label">Dominant income</div><div class="value" style="font-size:17px">{gc.income_band.mode().iat[0]}</div></div>
            <div class="prism-kpi"><div class="label">Top loyalty tier</div><div class="value" style="font-size:17px">{gc.loyalty_tier.mode().iat[0]}</div></div>
          </div>
          <div class="divider"></div>
          <div class="eyebrow">Top cities</div>
          <div class="rec-body">{', '.join(f'{c} ({v:.0%})' for c, v in cities.items())}</div>
        </div>""", unsafe_allow_html=True)

        prof = segmentation.segment_profile(full, core["tx"], core["prods"], seg)
        section("What they buy")
        cats = list(prof["top_categories"].items())
        fig = go.Figure(go.Bar(
            x=[v for _, v in cats], y=[c for c, _ in cats], orientation="h",
            marker_color="rgba(45,212,191,.6)"))
        fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Revenue (₹)")
        style_fig(fig, 240)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with right:
        prof = segmentation.segment_profile(full, core["tx"], core["prods"], seg)
        b = prof["behaviour"]
        section("How they behave")
        kpi_row([
            dict(label="Avg spend", value=f"₹{b['avg_spend']:,.0f}", sub="lifetime"),
            dict(label="Discount availed", value=f"{b['avg_discount']:.1%}", sub="avg per order"),
            dict(label="Buys on promo", value=f"{b['deal_purchase_share']:.0%}", sub="orders >5% off"),
            dict(label="Return rate", value=f"{b['return_rate']:.1%}"),
            dict(label="Frequency", value=f"{b['frequency']:.1f}", sub="orders / lifetime"),
            dict(label="Avg recency", value=f"{prof['avg_recency']:.0f}d"),
        ])

        e = core["seg_elastic"].set_index("segment").elasticity.get(seg, np.nan)
        sug = ("Flash sales & bundle offers — this segment amplifies discount response."
               if (pd.notna(e) and e < -1.2) else
               ("Loyalty perks & early access instead of discounts — price is not the lever."
                if pd.notna(e) and e > -0.8 else
                "Moderate targeted promotions; test incrementality before scaling."))
        section("Recommended promotion play")
        st.markdown(f"""
        <div class="rec-card">
          <div class="rec-body" style="font-size:13px">Segment price elasticity ≈
            <b style="color:{'#F87171' if (pd.notna(e) and e < -1.2) else '#2DD4BF'}">{e:.2f}</b> ·
            {sug}</div>
        </div>""", unsafe_allow_html=True)
