"""Data Explorer — inspect demo tables or upload your own CSV."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui_components import (ACCENT, MUTED, html_table, kpi_row, section, style_fig)

DEMO_TABLES = {
    "transactions.csv": "113k+ transactions · 24 months",
    "customers.csv": "10,000 customers with demographics",
    "products.csv": "300 products · 9 categories",
    "promotions.csv": "122 promotion events",
    "inventory.csv": "daily stock ledger · 300 SKUs",
    "market_signals.csv": "daily category-level market indices",
}


def render(core):
    tab_demo, tab_upload = st.tabs(["Demo datasets", "Upload your own CSV"])

    with tab_demo:
        name = st.selectbox("Dataset", list(DEMO_TABLES),
                            format_func=lambda n: f"{n} — {DEMO_TABLES[n]}")
        df = _load(core, name)
        _profile(df, name)

    with tab_upload:
        up = st.file_uploader("Upload a CSV to profile it (stays in your browser session)",
                              type=["csv"])
        if up is not None:
            try:
                df = pd.read_csv(up)
                _profile(df, up.name)
            except Exception as ex:
                st.error(f"Could not parse CSV: {ex}")
        else:
            st.markdown('<div class="prism-card rec-body" style="text-align:center;padding:30px">'
                        'Drop a CSV here to get instant profiling: schema, missing values, '
                        'distributions and a data-quality score.</div>', unsafe_allow_html=True)


@st.cache_data
def _load(core_hold, name):
    return pd.read_csv(f"data/{name}")


def _profile(df, name):
    n_rows, n_cols = df.shape
    missing_cells = int(df.isna().sum().sum())
    dup_rows = int(df.duplicated().sum())
    completeness = 1 - missing_cells / max(n_rows * n_cols, 1)
    uniqueness = 1 - dup_rows / max(n_rows, 1)
    quality = 100 * (0.6 * completeness + 0.4 * uniqueness)

    kpi_row([
        dict(label="Rows", value=f"{n_rows:,}"),
        dict(label="Columns", value=str(n_cols)),
        dict(label="Missing cells", value=f"{missing_cells:,}", sub=f"{(1-completeness):.1%} of all cells"),
        dict(label="Duplicate rows", value=f"{dup_rows:,}"),
        dict(label="Data quality score", value=f"{quality:.0f}/100",
             sub="completeness + uniqueness blend"),
    ])

    c1, c2 = st.columns([1.3, 1])
    with c1:
        section("Schema & sample")
        st.dataframe(df.head(50), width="stretch", height=320)
    with c2:
        section("Column profile")
        prof = pd.DataFrame({
            "column": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "missing%": (df.isna().mean() * 100).round(1).values,
            "unique": df.nunique().values,
        })
        html_table(prof, max_rows=20)

    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    if nums:
        section("Distributions")
        col = st.selectbox("Numeric column", nums)
        s = df[col].dropna()
        if len(s):
            fig = go.Figure(go.Histogram(x=s, nbinsx=40, marker_color="rgba(45,212,191,.6)"))
            stats = (f"mean {s.mean():,.1f} · median {s.median():,.1f} · "
                     f"p5 {s.quantile(.05):,.1f} · p95 {s.quantile(.95):,.1f}")
            style_fig(fig, 280, xaxis_title=col, yaxis_title="count")
            st.markdown(f'<span class="note">{stats}</span>', unsafe_allow_html=True)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.markdown("---")
    st.download_button("⬇ Download this dataset (CSV)", df.to_csv(index=False).encode(),
                       file_name=name if name.endswith(".csv") else f"{name}.csv",
                       mime="text/csv")
