"""PRISM UI design system — premium enterprise dark theme."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------- palette
BG = "#0B0E13"; PANEL = "#10141B"; CARD = "#141924"; CARD_HOVER = "#171E2B"
BORDER = "#222B3A"; TEXT = "#E7ECF5"; MUTED = "#8B95A9"; FAINT = "#5C6678"
ACCENT = "#2DD4BF"; ACCENT2 = "#7C8CF8"; GOOD = "#34D399"; WARN = "#F5B84C"; BAD = "#F87171"
CAT_COLORS = ["#2DD4BF", "#7C8CF8", "#F5B84C", "#F87171", "#34D399", "#60A5FA",
              "#E879F9", "#FB923C", "#A3E635"]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* {{ font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif; }}
html, body, [class*="css"] {{ color: {TEXT}; }}
.stApp {{ background: {BG}; }}
#MainMenu, header[data-testid="stHeader"], footer {{ visibility: hidden; }}
section[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid {BORDER}; }}
section[data-testid="stSidebar"] * {{ color: {TEXT}; }}
h1, h2, h3 {{ color: {TEXT}; letter-spacing: -0.02em; }}

/* ---------- cards ---------- */
.prism-card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px;
  padding: 18px 20px; margin-bottom: 4px; }}
.prism-kpi {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px;
  padding: 14px 16px; }}
.prism-kpi .label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.09em;
  color: {MUTED}; margin-bottom: 6px; white-space: nowrap; }}
.prism-kpi .value {{ font-size: 24px; font-weight: 650; color: {TEXT}; }}
.prism-kpi .delta {{ font-size: 12px; margin-top: 5px; font-weight: 600; }}
.up {{ color: {GOOD}; }} .down {{ color: {BAD}; }} .flat {{ color: {MUTED}; }}
.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px; margin: 8px 0 4px; }}

/* ---------- typography ---------- */
.prism-h {{ font-size: 15px; font-weight: 650; color: {TEXT}; display: flex; align-items: center;
  gap: 8px; margin: 0; }}
.prism-sub {{ font-size: 12.5px; color: {MUTED}; margin-top: 3px; }}
.accent-bar {{ display: inline-block; width: 4px; height: 16px; background: {ACCENT};
  border-radius: 2px; }}
.eyebrow {{ font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.14em;
  color: {ACCENT}; font-weight: 700; }}

/* ---------- badges ---------- */
.badge {{ display: inline-block; padding: 2.5px 9px; border-radius: 999px; font-size: 10.5px;
  font-weight: 650; letter-spacing: 0.03em; white-space: nowrap; }}
.b-critical {{ background: rgba(248,113,113,.14); color: {BAD}; border: 1px solid rgba(248,113,113,.35); }}
.b-high {{ background: rgba(245,184,76,.13); color: {WARN}; border: 1px solid rgba(245,184,76,.35); }}
.b-medium {{ background: rgba(124,140,248,.13); color: {ACCENT2}; border: 1px solid rgba(124,140,248,.35); }}
.b-low {{ background: rgba(139,149,169,.12); color: {MUTED}; border: 1px solid rgba(139,149,169,.3); }}
.b-good {{ background: rgba(52,211,153,.13); color: {GOOD}; border: 1px solid rgba(52,211,153,.35); }}
.b-accent {{ background: rgba(45,212,191,.12); color: {ACCENT}; border: 1px solid rgba(45,212,191,.35); }}

/* ---------- recommendation cards ---------- */
.rec-card {{ background: {CARD}; border: 1px solid {BORDER}; border-left: 3px solid {ACCENT};
  border-radius: 12px; padding: 14px 16px; margin-bottom: 10px; }}
.rec-title {{ font-size: 13.5px; font-weight: 650; color: {TEXT}; margin-bottom: 4px; }}
.rec-body {{ font-size: 12.5px; color: {MUTED}; line-height: 1.55; }}
.rec-why li {{ font-size: 12px; color: {MUTED}; margin: 3px 0; line-height: 1.45; }}
.rec-meta {{ display: flex; gap: 14px; margin-top: 8px; flex-wrap: wrap; }}
.meta-item {{ font-size: 11px; color: {FAINT}; }}
.meta-item b {{ color: {TEXT}; font-weight: 600; }}

/* ---------- tables ---------- */
.prism-table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
.prism-table th {{ text-align: left; padding: 8px 10px; color: {MUTED}; font-size: 10.5px;
  text-transform: uppercase; letter-spacing: 0.08em; border-bottom: 1px solid {BORDER};
  font-weight: 650; }}
.prism-table td {{ padding: 8px 10px; border-bottom: 1px solid rgba(34,43,58,.55); color: {TEXT}; }}
.prism-table tr:hover td {{ background: {CARD_HOVER}; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; }}

/* ---------- sidebar nav ---------- */
nav.prism-nav {{ display: flex; flex-direction: column; gap: 3px; margin-top: 8px; }}
.prism-nav button {{ all: unset; cursor: pointer; display: flex; align-items: center; gap: 10px;
  padding: 8.5px 12px; border-radius: 9px; font-size: 13.5px; color: {MUTED}; font-weight: 500;
  transition: all .15s; width: 100%; box-sizing: border-box; }}
.prism-nav button:hover {{ background: rgba(45,212,191,.07); color: {TEXT}; }}
.prism-nav button.active {{ background: rgba(45,212,191,.11); color: {ACCENT}; font-weight: 600;
  border: 1px solid rgba(45,212,191,.25); }}
.prism-nav .ico {{ width: 17px; text-align: center; opacity: .9; }}
.sidebar-foot {{ margin-top: 14px; border-top: 1px solid {BORDER}; padding-top: 10px;
  font-size: 10.5px; color: {FAINT}; line-height: 1.7; }}

/* logo */
.logo-wrap {{ display: flex; align-items: center; gap: 10px; padding: 4px 2px 12px; }}
.logo-mark {{ width: 34px; height: 34px; border-radius: 9px;
  background: linear-gradient(135deg, #2DD4BF 0%, #7C8CF8 100%);
  display: flex; align-items: center; justify-content: center; font-weight: 800;
  color: #0B0E13; font-size: 15px; }}
.logo-text {{ font-size: 19px; font-weight: 750; letter-spacing: 0.02em; color: {TEXT}; }}
.logo-sub {{ font-size: 9.5px; color: {FAINT}; letter-spacing: 0.05em; margin-top: -2px; }}

/* methodology flow */
.flow {{ display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin: 10px 0; }}
.flow .step {{ background: {CARD}; border: 1px solid {BORDER}; padding: 8px 14px;
  border-radius: 10px; font-size: 12px; font-weight: 600; color: {TEXT}; }}
.flow .arrow {{ color: {ACCENT}; font-size: 14px; }}
.note {{ font-size: 11px; color: {FAINT}; }}
.divider {{ height: 1px; background: {BORDER}; margin: 10px 0; }}
</style>
"""

PLOT_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=12, color=MUTED),
    margin=dict(l=10, r=10, t=36, b=10),
    xaxis=dict(gridcolor="rgba(255,255,255,.06)", zerolinecolor="rgba(255,255,255,.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,.06)", zerolinecolor="rgba(255,255,255,.1)"),
    legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)),
    hoverlabel=dict(bgcolor=CARD, bordercolor=BORDER, font=dict(color=TEXT)),
)


def style_fig(fig, height=320, **kw):
    lay = dict(PLOT_LAYOUT, height=height, **kw)
    fig.update_layout(**lay)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def fmt_inr(v, cr=False):
    if abs(v) >= 1e7 or cr:
        return f"₹{v/1e7:.1f} Cr"
    if abs(v) >= 1e5:
        return f"₹{v/1e5:.1f} L"
    if abs(v) >= 1e3:
        return f"₹{v/1e3:.1f}K"
    return f"₹{v:,.0f}"


def kpi_row(items: list[dict]):
    """items: [{label, value, delta, good_up=True}]"""
    cells = []
    for it in items:
        delta = it.get("delta")
        if delta is None:
            d_html = f'<div class="delta flat">{it.get("sub","")}</div>'
        else:
            good_up = it.get("good_up", True)
            cls = ("up" if (delta >= 0) == good_up else "down")
            sign = "+" if delta >= 0 else ""
            d_html = (f'<div class="delta {cls}">{sign}{delta*100:.1f}% '
                      f'<span class="flat" style="font-weight:400">vs prev</span></div>')
        cells.append(
            f'<div class="prism-kpi"><div class="label">{it["label"]}</div>'
            f'<div class="value">{it["value"]}</div>{d_html}</div>')
    st.markdown(f'<div class="kpi-grid">{"".join(cells)}</div>', unsafe_allow_html=True)


def section(title, sub=None):
    bar = '<span class="accent-bar"></span>'
    sub = f'<div class="prism-sub">{sub}</div>' if sub else ""
    st.markdown(f'<div style="margin:6px 0 10px">{bar}<span class="prism-h">{title}</span>{sub}</div>',
                unsafe_allow_html=True)


def band_badge(band: str):
    b = str(band)
    cls = {"Critical": "b-critical", "High": "b-high", "Medium": "b-medium",
           "Low": "b-low"}.get(b, "b-low")
    return f'<span class="badge {cls}">{b}</span>'


def conf_badge(c: int):
    cls = "b-good" if c >= 75 else ("b-medium" if c >= 60 else "b-low")
    return f'<span class="badge {cls}">{c}% confidence</span>'


def rec_card(rec: dict, show_kpis=True):
    why = "".join(f"<li>{w}</li>" for w in rec["why"])
    imp = rec.get("impact") or {}
    imp_html = ""
    if imp:
        imp_html = (f'<div class="meta-item">Expected units <b>{imp.get("units",0):,.0f}/qtr</b></div>'
                    f'<div class="meta-item">Revenue <b>{fmt_inr(imp.get("revenue",0))}</b></div>'
                    f'<div class="meta-item">Contribution <b>{fmt_inr(imp.get("contribution_margin",0))}</b></div>')
    kpis = ""
    if show_kpis and rec.get("kpis"):
        k = rec["kpis"]
        kpis = (f'<div class="meta-item">Elasticity <b>{k["elasticity"]}</b></div>'
                f'<div class="meta-item">Inv days <b>{k["inventory_days"]:.0f}</b></div>'
                f'<div class="meta-item">Margin <b>{k["margin_pct"]:.0%}</b></div>'
                f'<div class="meta-item">Comp gap <b>{k["competitor_gap"]:+.1%}</b></div>')
    st.markdown(f"""
    <div class="rec-card">
      <div class="rec-title">{rec['product_name']}
        <span class="badge b-accent">{rec['action']}</span>
        {band_badge(rec.get('opportunity_band','-'))}
        {conf_badge(rec['confidence'])}</div>
      <div class="rec-body">{rec['recommendation']}</div>
      <div class="divider"></div>
      <div class="eyebrow">Why this recommendation</div>
      <ul class="rec-why">{why}</ul>
      <div class="rec-meta">{imp_html}{kpis}</div>
    </div>""", unsafe_allow_html=True)


def html_table(df: pd.DataFrame, num_cols=(), formatters=None, max_rows=15):
    formatters = formatters or {}
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = []
    for _, r in df.head(max_rows).iterrows():
        tds = []
        for c in df.columns:
            v = r[c]
            f = formatters.get(c)
            txt = f(v) if f else (f"{v:,.0f}" if isinstance(v, (int, float)) and c in num_cols else str(v))
            cls = "num" if c in num_cols or f else ""
            tds.append(f'<td class="{cls}">{txt}</td>')
        rows.append(f"<tr>{''.join(tds)}</tr>")
    st.markdown(f'<div class="prism-card" style="padding:6px 12px"><table class="prism-table">'
                f'<thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>',
                unsafe_allow_html=True)


def empty_hint(msg):
    st.markdown(f'<div class="prism-card rec-body" style="text-align:center;padding:26px">{msg}</div>',
                unsafe_allow_html=True)
