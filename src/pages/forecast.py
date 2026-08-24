"""Demand Forecast — ML model transparency: metrics, importance, predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui_components import (ACCENT, ACCENT2, GOOD, MUTED, WARN, fmt_inr, html_table,
                               kpi_row, section, style_fig)


def render(core):
    dm, pm = core["dmetrics"], core["pmetrics"]

    section("Model status", "Temporal split — trained on first 80% of weeks, tested on the most recent 20%")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="prism-card">
          <div class="eyebrow">Model 1 · Demand forecasting</div>
          <div style="font-size:15px;font-weight:650;margin:6px 0 4px">XGBoost — weekly SKU panel</div>
          <div class="rec-body">Target: units per product-week · n={dm['n_train']:,} train / {dm['n_test']:,} test rows</div>
          <div class="divider"></div>
          <div class="kpi-grid">
            <div class="prism-kpi"><div class="label">R²</div><div class="value">{dm['xgb']['r2']:.3f}</div></div>
            <div class="prism-kpi"><div class="label">MAE</div><div class="value">{dm['xgb']['mae']:.2f}</div></div>
            <div class="prism-kpi"><div class="label">RMSE</div><div class="value">{dm['xgb']['rmse']:.2f}</div></div>
          </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="prism-card">
          <div class="eyebrow">Model 2 · Promotion response</div>
          <div style="font-size:15px;font-weight:650;margin:6px 0 4px">XGBoost — promo uplift ratio</div>
          <div class="rec-body">Target: actual / expected units during events · n={pm['n_obs']:,} product-event pairs</div>
          <div class="divider"></div>
          <div class="kpi-grid">
            <div class="prism-kpi"><div class="label">R²</div><div class="value">{pm['r2']:.3f}</div></div>
            <div class="prism-kpi"><div class="label">MAE</div><div class="value">{pm['mae']:.3f}</div></div>
            <div class="prism-kpi"><div class="label">RMSE</div><div class="value">{pm['rmse']:.3f}</div></div>
          </div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="prism-card">
          <div class="eyebrow">Model 1 · Baselines</div>
          <div style="font-size:15px;font-weight:650;margin:6px 0 4px">Linear vs Random Forest</div>
          <div class="rec-body">Both outperformed by gradient boosting — non-linear price × seasonality interactions matter.</div>
          <div class="divider"></div>
          <table class="prism-table">
            <tr><th>Model</th><th class="num">R²</th><th class="num">MAE</th><th class="num">RMSE</th></tr>
            <tr><td>Linear regression</td><td class="num">{dm['linear']['r2']:.3f}</td><td class="num">{dm['linear']['mae']:.2f}</td><td class="num">{dm['linear']['rmse']:.2f}</td></tr>
            <tr><td>Random forest</td><td class="num">{dm['random_forest']['r2']:.3f}</td><td class="num">{dm['random_forest']['mae']:.2f}</td><td class="num">{dm['random_forest']['rmse']:.2f}</td></tr>
            <tr><td><b>XGBoost</b></td><td class="num"><b>{dm['xgb']['r2']:.3f}</b></td><td class="num"><b>{dm['xgb']['mae']:.2f}</b></td><td class="num"><b>{dm['xgb']['rmse']:.2f}</b></td></tr>
          </table>
        </div>""", unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        section("What drives demand", "Feature importance — demand model")
        fi = list(dm["feature_importance"].items())[:12][::-1]
        labels = {"lag_units_1": "Units (lag 1w)", "roll4_units": "Rolling 4w avg", "price": "Price",
                  "disc": "Discount %", "search_interest": "Search interest", "seasonality": "Seasonality",
                  "market_demand": "Market demand idx", "lag_units_2": "Units (lag 2w)",
                  "inventory_days": "Inventory days", "competitor_gap_pct": "Competitor gap %",
                  "price_change_pct": "Price change %", "price_vs_list": "Price vs list",
                  "margin_pct": "Margin %", "closing_inventory": "Closing inventory",
                  "sentiment": "Economic sentiment", "stockout": "Stockout flag", "rating": "Product rating",
                  "lifecycle_code": "Lifecycle stage", "category_code": "Category", "week_num": "Week index"}
        fig = go.Figure(go.Bar(
            x=[v for _, v in fi], y=[labels.get(k, k) for k, _ in fi], orientation="h",
            marker_color="rgba(45,212,191,.65)"))
        style_fig(fig, 340)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        section("What drives promotion response", "Feature importance — promo model")
        fi2 = list(pm["feature_importance"].items())[:10][::-1]
        labels2 = {"discount_pct": "Discount depth", "elasticity": "Price elasticity",
                   "base_velocity": "Base velocity", "duration_days": "Duration",
                   "promo_code": "Promo type", "cat_code": "Category",
                   "campaign_cost": "Campaign cost", "target_deal_seekers": "Targets deal seekers",
                   "festive": "Festive period", "rating": "Product rating"}
        fig = go.Figure(go.Bar(
            x=[v for _, v in fi2], y=[labels2.get(k, k) for k, _ in fi2], orientation="h",
            marker_color="rgba(124,140,248,.6)"))
        style_fig(fig, 300)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with right:
        section("Forecast explorer", "8-week XGBoost forecast per product")
        prods = core["prods"].set_index("product_id")
        fc = core["forecast"]
        top = fc.groupby("product_id").forecast_units.sum().sort_values(ascending=False).head(60)
        pid = st.selectbox("Product", list(top.index),
                           format_func=lambda p: f"{prods.product_name[p]}")
        g = fc[fc.product_id == pid].sort_values("week")
        hist = core["panel"][core["panel"].product_id == pid].tail(16).sort_values("week")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.week, y=hist.units, mode="lines+markers",
                                 line=dict(color=MUTED, width=1.8), name="Actual"))
        fig.add_trace(go.Scatter(x=g.week, y=g.forecast_units, mode="lines+markers",
                                 line=dict(color=ACCENT, width=2.5, dash="dot"), name="Forecast"))
        fig.update_layout(yaxis_title="Units / week")
        style_fig(fig, 300)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        section("Category demand outlook", "Forecast units by category, next 8 weeks")
        fcc = fc.merge(core["prods"][["product_id", "category"]], on="product_id")
        cat_f = fcc.groupby(["category", "week"]).forecast_units.sum().reset_index()
        for i, (cat, gd) in enumerate(cat_f.groupby("category")):
            fig.add_trace(go.Scatter(x=gd.week, y=gd.forecast_units, mode="lines", name=cat,
                                     line=dict(color=["#2DD4BF", "#7C8CF8", "#F5B84C", "#F87171",
                                                      "#34D399", "#60A5FA", "#E879F9", "#FB923C",
                                                      "#A3E635"][i % 9], width=2)))
        style_fig(fig, 320, yaxis_title="Units / week")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
