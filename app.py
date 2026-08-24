"""PRISM — Pricing & Revenue Intelligence System (NOVA MART demo).

Run:  streamlit run app.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from src import ui_components as ui
from src.pages import (about, explorer, forecast, inventory_health, overview,
                       pricing_intel, promo_lab, recommendations, segments, simulator)

ROOT = Path(__file__).resolve().parent
DATA, CACHE = ROOT / "data", ROOT / "cache"

st.set_page_config(page_title="PRISM — Pricing & Revenue Intelligence", page_icon="◈",
                   layout="wide", initial_sidebar_state="expanded")
ui.inject_css()

PAGES = [
    ("overview", "◎", "Overview", "Revenue command center"),
    ("pricing", "⚖", "Pricing Intelligence", "Product pricing economics"),
    ("promolab", "⚗", "Promotion Lab", "Simulate & compare promotions"),
    ("segments", "👥", "Customer Segments", "RFM + behavioural segments"),
    ("forecast", "📈", "Demand Forecast", "ML demand model"),
    ("inventory", "📦", "Inventory Health", "Risk matrix & stock analytics"),
    ("recs", "⚡", "Product Recommendations", "Ranked pricing actions"),
    ("simulator", "🎚", "Scenario Simulator", "What-if modelling"),
    ("explorer", "🗃", "Data Explorer", "Datasets & quality"),
    ("about", "ⓘ", "About this Project", "Methodology & case study"),
]


@st.cache_resource(show_spinner="Loading analytics core …")
def load_core():
    with open(CACHE / "core.pkl", "rb") as f:
        core = pickle.load(f)
    core["tx"] = pd.read_csv(DATA / "transactions.csv", parse_dates=["transaction_date"])
    core["prods"] = pd.read_csv(DATA / "products.csv")
    core["cust"] = pd.read_csv(DATA / "customers.csv")
    core["promos"] = pd.read_csv(DATA / "promotions.csv", parse_dates=["start_date", "end_date"])
    core["inv"] = pd.read_csv(DATA / "inventory.csv", parse_dates=["date"])
    core["signals"] = pd.read_csv(DATA / "market_signals.csv", parse_dates=["date"])
    return core


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("""
    <div class="logo-wrap">
      <div class="logo-mark">◈</div>
      <div><div class="logo-text">PRISM</div>
      <div class="logo-sub">PRICING & REVENUE INTELLIGENCE</div></div>
    </div>""", unsafe_allow_html=True)
    st.caption("Turn customer behavior, market signals and inventory into smarter pricing decisions.")

    st.markdown('<div class="eyebrow" style="margin-bottom:2px">Workspace</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex;gap:8px;align-items:center;font-size:12px;color:#8B95A9">
      <span class="badge b-accent">NOVA MART</span>
      <span>Consumer Electronics · India</span>
    </div>""", unsafe_allow_html=True)

    nav = st.radio("Navigation", [p[0] for p in PAGES], label_visibility="collapsed",
                   format_func=lambda k: dict((p[0], p[2]) for p in PAGES)[k])

    titles = dict((p[0], (p[2], p[3])) for p in PAGES)
    t, s = titles[nav]
    st.markdown(f'<div class="sidebar-foot">', unsafe_allow_html=True)

try:
    core = load_core()
except FileNotFoundError:
    st.error("Analytics cache not found. Run `python3 pipeline.py` first.")
    st.stop()

dm = core["dmetrics"]["xgb"]; pm = core["pmetrics"]
with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-foot">
      <div class="eyebrow">Model status</div>
      <div>Demand model <span class="badge b-good">R² {dm['r2']:.2f}</span></div>
      <div>Promo response <span class="badge b-good">R² {pm['r2']:.2f}</span></div>
      <div class="eyebrow" style="margin-top:8px">Data</div>
      <div>Updated 22 Aug 2026 · 24 months</div>
      <div style="color:#5C6678">Synthetic demonstration dataset — created for analytical demonstration.</div>
    </div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------- router
st.markdown(f'<div style="margin-bottom:2px"><span class="eyebrow">{t} · NOVA MART</span></div>'
            f'<h1 style="font-size:26px;margin:2px 0 14px">{t}</h1>', unsafe_allow_html=True)

KW = dict(core=core)
if nav == "overview":
    overview.render(**KW)
elif nav == "pricing":
    pricing_intel.render(**KW)
elif nav == "promolab":
    promo_lab.render(**KW)
elif nav == "segments":
    segments.render(**KW)
elif nav == "forecast":
    forecast.render(**KW)
elif nav == "inventory":
    inventory_health.render(**KW)
elif nav == "recs":
    recommendations.render(**KW)
elif nav == "simulator":
    simulator.render(**KW)
elif nav == "explorer":
    explorer.render(**KW)
elif nav == "about":
    about.render(**KW)
