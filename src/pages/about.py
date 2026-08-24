"""About — case study & methodology."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.ui_components import (ACCENT, GOOD, MUTED, html_table, kpi_row, section)

FLOW = ["DATA", "ANALYSIS", "PREDICTION", "OPTIMIZATION", "DECISION"]


def render(core):
    st.markdown("""
    <div class="prism-card" style="border-left:3px solid #2DD4BF">
      <div class="eyebrow">Portfolio case study</div>
      <div style="font-size:17px;font-weight:700;margin:4px 0 8px">
        PRISM — Pricing & Revenue Intelligence System</div>
      <div class="rec-body">
        A decision-support product for a fictional premium omnichannel electronics retailer
        (NOVA MART, India). It converts 24 months of transaction, customer, inventory, promotion
        and market-signal data into concrete pricing actions — going beyond dashboards to
        optimizer-backed, explainable recommendations.
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="flow">' +
                '<span class="arrow">→</span>'.join(
                    f'<span class="step">{s}</span>' for s in FLOW) +
                '</div>', unsafe_allow_html=True)

    section("Business questions answered")
    st.markdown("""
    <div class="prism-card"><ul class="rec-why" style="columns:2;-webkit-columns:2;margin:4px 0">
      <li>Which products <b>should</b> be discounted — and by how much?</li>
      <li>Which products must <b>never</b> be discounted (inelastic, thin margin)?</li>
      <li>Which customer segments should receive promotions?</li>
      <li>Which discounts lift revenue but destroy contribution margin?</li>
      <li>Where are we over/under-stocked relative to demand?</li>
      <li>Where are competitors structurally underpricing us?</li>
      <li>What is the expected incremental revenue & margin of an action?</li>
      <li>What should the business do <b>next</b> — ranked by opportunity?</li>
    </ul></div>""", unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        section("Data architecture", "Six relational tables · synthetic, statistically-coherent")
        st.markdown("""
        <div class="prism-card"><table class="prism-table">
          <tr><th>Table</th><th>Rows</th><th>Role</th></tr>
          <tr><td>transactions.csv</td><td class="num">113k+</td><td>24 months of order lines</td></tr>
          <tr><td>customers.csv</td><td class="num">10,000</td><td>demographics + acquisition</td></tr>
          <tr><td>products.csv</td><td class="num">300</td><td>catalogue, cost, competitor price, stock</td></tr>
          <tr><td>promotions.csv</td><td class="num">122</td><td>8 promo types w/ costs & targets</td></tr>
          <tr><td>inventory.csv</td><td class="num">218k</td><td>daily stock ledger, stockouts</td></tr>
          <tr><td>market_signals.csv</td><td class="num">6.5k</td><td>competitor, demand, search, sentiment</td></tr>
        </table></div>""", unsafe_allow_html=True)

        section("Relationships built into the data (not random noise)")
        st.markdown(f"""
        <div class="prism-card"><ul class="rec-why">
          <li>Demand follows constant-elasticity response Q = A·P<sup>−E</sup>, with category-level
              designed elasticities (Accessories ≈ 2.0 … Cameras ≈ 0.8)</li>
          <li>Customer archetypes modulate discount response — Deal Seekers buy on promo,
              Premium Loyalists don't</li>
          <li>Indian festive calendar drives seasonality (Oct–Nov peak, Republic Day, Aug 15)</li>
          <li>Inventory responds to sales via (s,S) replenishment with lead times — stockouts cap demand</li>
          <li>Competitor price gaps shift purchase probability</li>
          <li>Mid-2025 economic dip is visible in market signals and sales</li>
        </ul></div>""", unsafe_allow_html=True)

    with right:
        section("Validation — designed vs recovered elasticity",
                "The econometrics recover the causal structure baked into the simulation")
        el = core["elastic"].dropna(subset=["elasticity"])
        import json
        meta = json.loads(open("data/meta.json").read())
        el["designed"] = el.product_id.map(meta["designed_elasticity"])
        agg = el.groupby("category").agg(designed=("designed", "median"),
                                         recovered=("elasticity", "median"),
                                         significant=("t_stat", lambda s: (s.abs() > 1.96).mean())).round(2)
        agg["significant"] = (agg.significant * 100).round(0).astype(int).astype(str) + "%"
        html_table(agg.reset_index(), max_rows=10)

        section("Segment recovery", "Unsupervised pipeline vs designed behavioural archetypes")
        ct = core["ctab"]
        diag = (ct.max(axis=1) / ct.sum(axis=1) * 100).round(0).astype(int)
        html_table(pd.DataFrame({"designed archetype": ct.index,
                                 "recovered as": ct.idxmax(axis=1).values,
                                 "match rate": [f"{v}%" for v in diag.values]}), max_rows=10)

        section("Models")
        dm, pm = core["dmetrics"], core["pmetrics"]
        st.markdown(f"""
        <div class="prism-card"><table class="prism-table">
          <tr><th>Model</th><th>Algorithm</th><th class="num">R²</th><th class="num">MAE</th></tr>
          <tr><td>Demand forecast</td><td>XGBoost (weekly SKU panel)</td>
              <td class="num">{dm['xgb']['r2']:.2f}</td><td class="num">{dm['xgb']['mae']:.2f} units</td></tr>
          <tr><td>Linear baseline</td><td>OLS</td>
              <td class="num">{dm['linear']['r2']:.2f}</td><td class="num">{dm['linear']['mae']:.2f}</td></tr>
          <tr><td>Promotion response</td><td>XGBoost (uplift ratio)</td>
              <td class="num">{pm['r2']:.2f}</td><td class="num">{pm['mae']:.2f}x</td></tr>
        </table></div>""", unsafe_allow_html=True)

    section("Decision engine — transparent rule cascade")
    st.markdown("""
    <div class="prism-card"><ul class="rec-why">
      <li><b>R1 Margin protection:</b> elastic demand + thin margin → never discount (units ≠ value)</li>
      <li><b>R2 Overstock monetisation:</b> high inventory days + elastic demand → targeted promotion for Deal Seekers</li>
      <li><b>R3 Competitive defence:</b> competitor gap &gt; 6% with margin headroom → controlled price reduction (floored at 85% of competitor)</li>
      <li><b>R4 Availability first:</b> stockout risk + inelastic demand → hold price, replenish</li>
      <li><b>R5 Price discipline:</b> inelastic + healthy stock → maintain price</li>
      <li><b>R6 Experimentation:</b> elastic + healthy margin + balanced stock → small measured test</li>
    </ul></div>""", unsafe_allow_html=True)

    section("Key business findings surfaced in this demo")
    pe = core["promo_eff"]
    st.markdown(f"""
    <div class="prism-card"><ul class="rec-why">
      <li>Median promotion uplift is <b>+{pe.uplift.median():.0%}</b>, but median promo
          <b>ROI is {pe.roi.median():.1f}x</b> — over half of events destroy contribution margin
          (discount dependency is real and measurable).</li>
      <li>Flash Sales deliver the best uplift; Loyalty Offers deliver the worst ROI per rupee.</li>
      <li>Premium Loyalists generate ~{core['seg_summary'].iloc[0].revenue_share:.0%} of revenue with
          ~{core['seg_summary'].iloc[0].avg_discount:.0%} average discount — deep discounts aimed at them are wasted.</li>
      <li>Overstocked, high-margin SKUs are the optimizer's best targets — margin-funded promotions,
          not blanket sitewide sales.</li>
    </ul></div>""", unsafe_allow_html=True)

    section("Limitations & honest caveats")
    st.markdown("""
    <div class="prism-card"><ul class="rec-why">
      <li>Data is synthetic. Causal structure is designed-in, so estimates are cleaner than messy
          real-world data would allow; recovered elasticities are attenuated ~20–40% by count noise
          (a real phenomenon in SKU-level econometrics).</li>
      <li>Elasticities assume constant-response within observed price ranges; extrapolation beyond
          ±30% is unreliable.</li>
      <li>Promotion cannibalisation & halo effects are estimated from aggregate category dips, not
          causal designs (no geo-holdouts in this dataset).</li>
      <li>Forecast assumes current prices and no shocks; confidence intervals are model-based.</li>
      <li>Demographics are used only in aggregate for segmentation — no individual targeting logic
          is exported.</li>
    </ul></div>""", unsafe_allow_html=True)

    st.markdown('<div class="note">Synthetic demonstration dataset — created for analytical '
                'demonstration. NOVA MART is fictional; no real company data is used.</div>',
                unsafe_allow_html=True)
