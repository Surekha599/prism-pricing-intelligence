# PRISM — Pricing & Revenue Intelligence System

> Turn customer behavior, market signals and inventory into smarter pricing decisions.

**PRISM** is a portfolio-grade product-analytics & decisioning platform for a fictional premium
omnichannel electronics retailer, **NOVA MART** (India). It is not a dashboard — it is a
decision-support product: an optimizer plus a transparent decision engine that outputs *ranked,
explainable pricing actions* for every SKU.

**📄 Full case study:** [`docs/PRISM_Case_Study.html`](docs/PRISM_Case_Study.html) — methodology, 8 figures, model validation and business impact, all computed from the actual dataset.

> **Data disclosure:** Synthetic demonstration dataset — created for analytical demonstration.
> NOVA MART is fictional. No real company data is used. The generator (`src/data_generator.py`)
> builds *causal* relationships by design, so every downstream estimate is earned from raw data.

---

## What goes in → what comes out

**Inputs** (6 relational CSVs, generated deterministically by `src/data_generator.py`, seed 42):

| Table | Rows | Contents |
|---|---|---|
| `transactions.csv` | 113k+ | 24 months of order lines: price, discount, promo, channel, returns |
| `customers.csv` | 10,000 | 15 Indian cities, demographics, loyalty tiers |
| `products.csv` | 300 | 9 categories, cost / list / current / competitor price, stock |
| `promotions.csv` | 122 | 8 promo types with depth, duration, cost, target segment |
| `inventory.csv` | 218k | daily stock ledger with replenishment & stockouts |
| `market_signals.csv` | 6.5k | daily competitor price, demand, search, sentiment indices |

**Outputs** (all reproducible via `python3 pipeline.py`):

| Output | What it is |
|---|---|
| Price elasticity per SKU | log-log estimation with CIs, seasonal controls |
| Promotion effectiveness | uplift, incremental units/revenue/margin, ROI, cannibalisation, post-dips |
| Customer segments | RFM + KMeans → 7 named behavioural segments with elasticity |
| Demand forecast | XGBoost weekly SKU model (R² ≈ 0.60, temporal split) + baselines |
| Promotion response model | XGBoost uplift model on 9.3k product×event pairs (R² ≈ 0.42) |
| Ranked pricing actions | optimizer-validated recommendation, why, expected impact, confidence |
| Opportunity scores | 0–100 composite: elasticity × inventory × margin × competitor × demand |

**Headline findings:** median promo uplift **+24%** but median ROI **−2.1x** (120 of 122 events
loss-making) · **₹31.5L/quarter** of campaign spend redirectable · **₹8.4 Cr** working capital
at risk in overstocked SKUs · Premium Loyalists drive ~63% of revenue at ~5% discount.

---

## Architecture

```
Data Layer (6 relational CSVs)          src/data_generator.py
   ↓
Econometrics & analytics                src/analytics.py
   elasticity · promo effectiveness · inventory
   ↓
Segmentation (RFM + KMeans)             src/segmentation.py
   ↓
ML models (XGBoost)                     src/models.py
   demand forecast · promotion response
   ↓
Optimization + decision engine          src/optimizer.py
   constrained margin optimizer · opportunity score · rule cascade
   ↓
Streamlit UI (10 pages)                 app.py + src/pages/*
```

The dataset is *statistically coherent by construction*: demand follows a constant-elasticity
response `Q = A·P^(-E)` with category-level designed elasticities, customer archetypes modulate
discount response, the Indian festive calendar drives seasonality (Oct–Nov peak), inventory
reacts to sales with (s,S) replenishment and stockouts, and competitor gaps shift demand.
The About page (and the case study) include a **designed-vs-recovered validation** — the
econometrics recover the causal structure from transactions alone.

## The decision engine

An optimizer with the objective **maximise incremental contribution margin**, subject to:
≥8% margin floor · ≤30% discount · price ≥85% of competitor · inventory caps · campaign budget.
Segment-targeted promotions discount only the covered share of demand (price discrimination),
using uplifts estimated from historical events; clearance-risk stock is valued against a
hold-then-clear-at-−35% counterfactual. A transparent 6-rule cascade turns this into
Recommendation / Why / Expected impact / Confidence / KPIs.

## Quick start

```bash
pip install -r requirements.txt
python3 pipeline.py        # data → analytics → models → optimizer cache (~30 s, deterministic)
streamlit run app.py       # open http://localhost:8501
```

Optional:
```bash
python3 pipeline.py --force     # regenerate a fresh dataset (new random draw under seed 42)
python3 make_report.py          # rebuild the case-study report from current results
python3 tests/smoke_test.py     # headless render-test of all 10 pages
```

## Project structure

```
prism/
├── app.py                  # Streamlit entry + navigation
├── pipeline.py             # one-command build: data → analytics → ML → optimization
├── make_report.py          # regenerates the case study from live results
├── requirements.txt
├── src/
│   ├── config.py           # business constants (categories, segments, seasonality)
│   ├── data_generator.py   # synthetic data engine (causal by design)
│   ├── analytics.py        # revenue, elasticity, promotion effectiveness, inventory
│   ├── segmentation.py     # RFM + KMeans → 7 business segments
│   ├── models.py           # XGBoost demand & promo-response models
│   ├── optimizer.py        # margin optimizer, opportunity score, decision engine
│   ├── ui_components.py    # design system
│   └── pages/              # 10 UI page modules
├── tests/smoke_test.py     # AppTest-based page render tests
├── docs/PRISM_Case_Study.html
└── data/ cache/ models/    # generated artifacts (gitignored; pipeline.py recreates them)
```

## Stack

Python · Pandas · NumPy · SciPy · scikit-learn · XGBoost · Plotly · Streamlit

## License

MIT — see [LICENSE](LICENSE).
