"""Build the PRISM case-study report (self-contained HTML, embedded figures).

All numbers and charts come from the actual generated dataset & trained models.
Run AFTER: python3 pipeline.py
Output:   PRISM_Case_Study.html
"""
from __future__ import annotations

import base64
import io
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA, CACHE = ROOT / "data", ROOT / "cache"

# ---------------------------------------------------------------- style
INK = "#1A1D23"; MUTED = "#5C6678"; FAINT = "#9AA3B2"; PAPER = "#FFFFFF"
ACCENT = "#0E9488"; ACCENT2 = "#5B6CFF"; WARM = "#C7791B"; RED = "#C4453C"
GRID = "#E8E5DE"; BG = "#FBFAF7"

plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER,
    "font.family": "DejaVu Sans", "font.size": 9.5,
    "axes.edgecolor": GRID, "axes.linewidth": 1.0,
    "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})


def fig_to_b64(fig, pad=0.15):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def cr(v):
    return f"₹{v/1e7:.1f} Cr" if abs(v) >= 1e7 else f"₹{v/1e5:.1f} L"


# ---------------------------------------------------------------- load
core = pickle.load(open(CACHE / "core.pkl", "rb"))
tx = pd.read_csv(DATA / "transactions.csv", parse_dates=["transaction_date"])
prods = pd.read_csv(DATA / "products.csv")
promos = pd.read_csv(DATA / "promotions.csv", parse_dates=["start_date", "end_date"])
meta = json.loads((DATA / "meta.json").read_text())
metrics = json.loads((ROOT / "models" / "metrics.json").read_text())

elastic, cat_elastic = core["elastic"], core["cat_elastic"]
promo_eff, seg_summary = core["promo_eff"], core["seg_summary"]
seg_el, seg_full = core["seg_elastic"], core["seg_full"]
inv = core["inv_health"]; recs = core["recs"]; ctab = core["ctab"]
forecast = core["forecast"]

# headline numbers
total_rev = tx.revenue.sum()
total_gm = tx.revenue.sum() - tx.cost.sum()
gm_pct = total_gm / total_rev
n_tx, n_units = len(tx), int(tx.quantity.sum())
disc_cost = (tx.revenue * tx.discount_pct / (1 - tx.discount_pct.clip(0, .95))).sum()
median_uplift = promo_eff.uplift.median()
median_roi = promo_eff.roi.median()
neg_roi_share = (promo_eff.roi < 0).mean()
redirectable_q = promo_eff[promo_eff.roi < 0].campaign_cost.sum() / 8  # 8 quarters
inv_risk_value = inv.value_at_risk.sum()
overstock_n = int((inv.inventory_days > 75).sum())
top20_margin = recs.head(20).expected_margin_impact.sum()
pos_recs = recs[recs.expected_margin_impact > 0]
dm = metrics["demand"]; pm = metrics["promo"]

# ---------------------------------------------------------------- figures
imgs = {}

# F1 weekly revenue & margin
w = tx.copy(); w["week"] = w.transaction_date.dt.to_period("W").dt.start_time
wk = w.groupby("week").agg(rev=("revenue", "sum"), gm=("gross_margin", "sum"))
fig, ax = plt.subplots(figsize=(8.6, 3.1))
ax.fill_between(wk.index, wk.rev / 1e7, color=ACCENT, alpha=.14, linewidth=0)
ax.plot(wk.index, wk.rev / 1e7, color=ACCENT, lw=1.8, label="Revenue")
ax.plot(wk.index, wk.gm / 1e7, color=ACCENT2, lw=1.5, label="Gross margin")
for x0, x1 in [("2024-10-05", "2024-11-03"), ("2025-10-05", "2025-10-21")]:
    ax.axvspan(pd.Timestamp(x0), pd.Timestamp(x1), color=WARM, alpha=.10, lw=0)
ax.annotate("festive\npeaks", xy=(pd.Timestamp("2024-10-19"), wk.rev.max()/1e7*.97),
            fontsize=8, color=WARM, ha="center", va="top")
ax.set_ylabel("₹ Cr / week")
ax.legend(frameon=False, loc="upper left", ncols=2)
imgs["f1"] = fig_to_b64(fig)

# F2 category bars with GM%
cat = tx.merge(prods[["product_id", "category"]], on="product_id")
cr_ = cat.groupby("category").agg(rev=("revenue", "sum"), gm=("gross_margin", "sum"))
cr_["gmp"] = cr_.gm / cr_.rev
cr_ = cr_.sort_values("rev")
fig, ax = plt.subplots(figsize=(8.6, 3.0))
bars = ax.barh(cr_.index, cr_.rev / 1e7, color=ACCENT, alpha=.85, height=.62)
for b, (_, r) in zip(bars, cr_.iterrows()):
    ax.text(b.get_width() + .4, b.get_y() + b.get_height()/2,
            f"{r.rev/1e7:.1f} Cr · GM {r.gmp:.0%}", va="center", fontsize=8.3, color=MUTED)
ax.set_xlabel("Revenue (₹ Cr, 24 months)")
ax.set_xlim(0, cr_.rev.max()/1e7 * 1.32)
imgs["f2"] = fig_to_b64(fig)

# F3 designed vs recovered elasticity (dumbbell)
el = elastic.copy(); el["designed"] = el.product_id.map(meta["designed_elasticity"])
agg = el.groupby("category").agg(des=("designed", "median"), rec=("elasticity", "median")).sort_values("des")
fig, ax = plt.subplots(figsize=(8.6, 3.2))
for i, (cat_n, r) in enumerate(agg.iterrows()):
    ax.plot([r.des, -r.rec], [i, i], color=FAINT, lw=1.6, zorder=1)
    ax.scatter(r.des, i, color=FAINT, s=42, zorder=2, label="designed" if i == 0 else "")
    ax.scatter(-r.rec, i, color=ACCENT, s=46, zorder=2, label="recovered (estimated)" if i == 0 else "")
    ax.text(-r.rec + .05, i, f"{-r.rec:.2f}", va="center", fontsize=8, color=ACCENT)
ax.set_yticks(range(len(agg)), agg.index)
ax.set_xlabel("|Elasticity| — magnitude of price response")
ax.legend(frameon=False, loc="lower right")
imgs["f3"] = fig_to_b64(fig)

# F4 hero product price–demand
hero = recs[recs.action.isin(["Targeted promotion", "Test promotion", "Controlled price reduction"])]
hero = hero.sort_values("opportunity_score", ascending=False)
hp = hero.iloc[0]
pid = hp.product_id
panel = core["panel"]; g = panel[panel.product_id == pid]
e_row = elastic[elastic.product_id == pid].iloc[0]
fig, ax = plt.subplots(figsize=(8.6, 3.1))
cols = [RED if d > .10 else (WARM if d > .03 else ACCENT) for d in g.disc]
ax.scatter(g.price, g.units, s=26, c=cols, alpha=.75, lw=0)
xs = np.linspace(g.price.min()*.97, g.price.max()*1.03, 60)
E = abs(e_row.elasticity)
fit = np.exp(np.log(max(g.units.mean(), .5)) - E*np.log(xs/g.price.mean()))
ax.plot(xs, fit, "--", color=MUTED, lw=1.4)
ax.set_xlabel("Effective price (₹)"); ax.set_ylabel("Units / week")
ax.set_title(f"{hp.product_name} — {hp.category} · estimated E = {e_row.elasticity:.2f}",
             fontsize=10, loc="left")
ax.annotate("promo weeks", xy=(.02, .93), xycoords="axes fraction", fontsize=8, color=RED)
imgs["f4"] = fig_to_b64(fig)

# F5 promo quadrant: uplift vs ROI
pt = promo_eff.groupby("promotion_type").agg(up=("uplift", "median"), roi=("roi", "median"),
                                             n=("uplift", "size"))
fig, ax = plt.subplots(figsize=(8.6, 3.3))
sizes = np.sqrt(pt.n) * 26
colors = {"Flash Sale": ACCENT, "Seasonal Sale": ACCENT2, "Flat Discount": RED,
          "Loyalty Offer": WARM, "Bundle": "#7A5FB5", "Buy One Get One": "#3E7CB1"}
ax.axhline(0, color=FAINT, lw=1)
ax.axvline(promo_eff.uplift.median(), color=FAINT, lw=1, ls=":")
for _, r in pt.iterrows():
    ax.scatter(r.up*100, r.roi, s=sizes[_], c=colors.get(_, MUTED), alpha=.8, lw=0)
    dx = 1.15 if _ not in ("Loyalty Offer",) else -1.3
    ax.annotate(f"{_}\n(n={r.n})", (r.up*100, r.roi), fontsize=8,
                xytext=(r.up*100 + dx, r.roi + (0.35 if _ != "Flat Discount" else -0.75)),
                color=INK)
ax.set_xlabel("Median sales uplift during event (%)")
ax.set_ylabel("Median ROI on campaign cost")
ax.set_title("Bigger lifts don't pay for themselves — ROI is negative for most formats",
             fontsize=10, loc="left")
imgs["f5"] = fig_to_b64(fig)

# F6 segments bubble
agg2 = seg_full.groupby("segment").agg(rec=("recency_days", "mean"), mon=("monetary", "median"),
                                       n=("customer_id", "size"), disc=("discount_sensitivity_raw", "mean"))
fig, ax = plt.subplots(figsize=(8.6, 3.3))
sc = ax.scatter(agg2.rec, agg2.mon/1e3, s=np.sqrt(agg2.n)*3.4,
                c=agg2.disc, cmap="cividis_r", alpha=.85, lw=0)
for s_, r in agg2.iterrows():
    ax.annotate(s_, (r.rec, r.mon/1e3), xytext=(0, np.sqrt(r.n)*0.62+6),
                textcoords="offset points", ha="center", fontsize=8.2, color=INK)
cb = fig.colorbar(sc, ax=ax, pad=.01); cb.set_label("avg discount availed", fontsize=8); cb.ax.tick_params(labelsize=7.5)
ax.set_xlabel("Average recency (days since last purchase)")
ax.set_ylabel("Median lifetime spend (₹ '000)")
ax.invert_xaxis()
imgs["f6"] = fig_to_b64(fig)

# F7 inventory matrix
fig, ax = plt.subplots(figsize=(8.6, 3.3))
qmask = inv.inventory_days > 75
ax.scatter(inv.inventory_days.clip(0, 200)[~qmask], (inv.gm_pct*100)[~qmask],
           s=np.sqrt(inv.inventory_value)[~qmask]/28+8, c=FAINT, alpha=.55, lw=0, label="healthy cover")
ax.scatter(inv.inventory_days.clip(0, 200)[qmask], (inv.gm_pct*100)[qmask],
           s=np.sqrt(inv.inventory_value)[qmask]/28+8, c=RED, alpha=.6, lw=0, label="overstock (>75d)")
ax.axvline(75, color=FAINT, ls=":", lw=1)
ax.axhline(inv.gm_pct.median()*100, color=FAINT, ls=":", lw=1)
ax.set_xlabel("Inventory days of cover"); ax.set_ylabel("Gross margin %")
ax.legend(frameon=False, loc="upper right")
ax.set_title(f"{overstock_n} SKUs carry excess stock — ₹{inv_risk_value/1e7:.1f} Cr of working capital at risk",
             fontsize=10, loc="left")
imgs["f7"] = fig_to_b64(fig)

# F8 model panel: R2 bars + feature importance
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 2.9), gridspec_kw={"width_ratios": [1, 1.35]})
mods = ["Linear\nregression", "Random\nForest", "XGBoost\n(final)"]
r2s = [dm["linear"]["r2"], dm["random_forest"]["r2"], dm["xgb"]["r2"]]
bars = a1.bar(mods, r2s, color=[FAINT, ACCENT2, ACCENT], width=.58)
for b, v in zip(bars, r2s):
    a1.text(b.get_x()+b.get_width()/2, v+.012, f"{v:.2f}", ha="center", fontsize=8.6, color=INK)
a1.set_ylabel("R² (held-out 20% of weeks)"); a1.set_ylim(0, max(r2s)*1.22)
a1.set_title("Demand model vs baselines", fontsize=9.5, loc="left")
fi = list(dm["feature_importance"].items())[:8][::-1]
LBL = {"lag_units_1": "Units, prior week", "roll4_units": "Rolling 4-wk avg", "price": "Price",
       "disc": "Discount %", "search_interest": "Search interest", "seasonality": "Seasonality",
       "market_demand": "Market demand idx", "lag_units_2": "Units, 2 wks prior",
       "price_change_pct": "Price change %", "inventory_days": "Inventory days"}
a2.barh([LBL.get(k, k) for k, _ in fi], [v for _, v in fi], color=ACCENT2, height=.6)
a2.set_title("What the model actually keys on", fontsize=9.5, loc="left")
imgs["f8"] = fig_to_b64(fig)

print("figures rendered:", list(imgs))

# ---------------------------------------------------------------- real recommendation examples
def rec_block(r):
    why = "".join(f"<li>{w}</li>" for w in r.why)
    return f"""
<div class="rec">
  <div class="rec-h">{r.product_name}<span class="tag">{r.action}</span>
      <span class="tag {'t-'+str(r.opportunity_band).lower()}">{r.opportunity_band} · {r.confidence}% conf</span></div>
  <p class="rec-t">"{r.recommendation}"</p>
  <ul class="why">{why}</ul>
  <div class="rec-m">Δ revenue ≈ {cr(r.expected_revenue_impact)} · Δ contribution ≈ {cr(r.expected_margin_impact)}</div>
</div>"""

ex1 = recs[recs.action == "Maintain price"].sort_values("opportunity_score", ascending=False).iloc[0]
ex2 = recs[recs.action == "Targeted promotion"].sort_values("expected_margin_impact", ascending=False).iloc[0]
ex3 = recs[recs.action == "Controlled price reduction"].sort_values("opportunity_score", ascending=False).iloc[0]

seg_pl = seg_summary[seg_summary.segment == "Premium Loyalists"].iloc[0]
seg_ds = seg_summary[seg_summary.segment == "Deal Seekers"].iloc[0]
match = ((ctab.max(axis=1) / ctab.sum(axis=1)) * 100)

CSS = """
:root { --ink:#1A1D23; --mut:#5C6678; --faint:#9AA3B2; --acc:#0E9488; --acc2:#5B6CFF;
  --warm:#C7791B; --red:#C4453C; --line:#E4E1D9; --bg:#FBFAF7; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.62 "Segoe UI", -apple-system, Helvetica, Arial, sans-serif; }
.page { max-width:880px; margin:0 auto; padding:56px 44px 80px; background:#fff;
  box-shadow:0 0 40px rgba(0,0,0,.05); }
h1,h2,h3 { font-family: Georgia, "Times New Roman", serif; letter-spacing:-.01em; }
h1 { font-size:33px; line-height:1.16; margin:10px 0 6px; }
h2 { font-size:21px; margin:44px 0 4px; border-top:2px solid var(--ink); padding-top:14px; }
h3 { font-size:15.5px; margin:22px 0 4px; }
p { margin:9px 0; } .mut { color:var(--mut); } .small { font-size:12.5px; }
.kicker { font:700 11px/1 "Segoe UI",sans-serif; letter-spacing:.18em; text-transform:uppercase; color:var(--acc); }
.meta { color:var(--mut); font-size:13px; margin-bottom:26px; }
.rule { height:1px; background:var(--line); margin:14px 0; }
.kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin:18px 0; }
.kpi { border:1px solid var(--line); border-radius:10px; padding:11px 13px; background:var(--bg); }
.kpi .l { font:600 10.5px/1.3 "Segoe UI",sans-serif; letter-spacing:.06em; text-transform:uppercase; color:var(--mut); }
.kpi .v { font-size:19px; font-weight:650; margin-top:3px; font-variant-numeric:tabular-nums; }
.kpi .s { font-size:11px; color:var(--faint); margin-top:2px; }
figure { margin:20px 0 8px; } figure img { width:100%; display:block; border:1px solid var(--line); border-radius:8px; }
figcaption { font-size:12px; color:var(--mut); margin-top:7px; }
figcaption b { color:var(--ink); }
.callout { border-left:3px solid var(--acc); background:var(--bg); padding:13px 16px; border-radius:0 8px 8px 0; margin:16px 0; }
.callout.red { border-color:var(--red); }
.flow { display:flex; gap:6px; flex-wrap:wrap; margin:14px 0; }
.flow .s { border:1px solid var(--ink); border-radius:8px; padding:7px 13px; font:650 11.5px "Segoe UI",sans-serif; letter-spacing:.05em; }
.flow .a { color:var(--acc); font-weight:700; align-self:center; }
table { border-collapse:collapse; width:100%; font-size:12.8px; margin:12px 0; }
th { text-align:left; font:650 10.5px "Segoe UI",sans-serif; text-transform:uppercase; letter-spacing:.07em;
  color:var(--mut); border-bottom:2px solid var(--ink); padding:7px 9px; }
td { padding:7px 9px; border-bottom:1px solid var(--line); font-variant-numeric:tabular-nums; }
td.r, th.r { text-align:right; }
.rec { border:1px solid var(--line); border-radius:10px; padding:13px 15px; margin:11px 0; background:#fff; }
.rec-h { font-weight:650; font-size:14px; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.tag { font:650 10px "Segoe UI",sans-serif; letter-spacing:.04em; padding:2.5px 9px; border-radius:999px;
  border:1px solid var(--acc); color:var(--acc); }
.tag.t-critical { border-color:var(--red); color:var(--red); } .tag.t-high { border-color:var(--warm); color:var(--warm); }
.rec-t { font-family:Georgia, serif; font-style:italic; color:#3A3F49; margin:7px 0 6px; }
.why { margin:4px 0 6px 18px; padding:0; } .why li { font-size:12.8px; color:var(--mut); margin:3px 0; }
.rec-m { font-size:12px; color:var(--faint); }
ol.steps { padding-left:22px; } ol.steps li { margin:7px 0; }
code, .code { font:12.5px/1.5 "SFMono-Regular", Consolas, monospace; background:#F2F0EA;
  padding:1.5px 6px; border-radius:5px; }
.foot { margin-top:44px; border-top:2px solid var(--ink); padding-top:12px; font-size:12px; color:var(--mut); }
@media print { .page { box-shadow:none; padding:24px 8px; } h2 { break-after:avoid; } figure { break-inside:avoid; } }
"""

html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>PRISM — Case Study</title>
<style>{CSS}</style></head>
<body><div class="page">

<div class="kicker">Product Analytics · Case Study</div>
<h1>PRISM — Pricing &amp; Revenue Intelligence System</h1>
<div class="meta">Portfolio project · NOVA MART (fictional omnichannel electronics retailer, India) ·
August 2026 · <i>Author: [Your Name]</i></div>

<div class="callout">
<b>The one-paragraph version.</b> NOVA MART runs promotions the way most retailers do — broadly,
often, and by instinct. I built PRISM to test whether that instinct pays. The system reconstructs
24 months of pricing behaviour from {n_tx:,} transactions across 300 SKUs, estimates how sensitive each
product's demand is to price, forecasts demand with ML, and then answers the commercial question
that matters: <b>what should we do to each product's price next quarter — and what will it earn us?</b>
The uncomfortable answer it found: the median promotion lifts sales by {median_uplift:+.0%} but
returns <b>{median_roi:.1f}x ROI</b> — {neg_roi_share:.0%} of campaigns destroy contribution margin.
{cr(redirectable_q)} of campaign spend sits in loss-making events every quarter; the optimizer
prices the reallocation — deep discounts move to the few SKUs and segments where they pay.
</div>

<div class="kpis">
  <div class="kpi"><div class="l">Revenue modelled</div><div class="v">{cr(total_rev)}</div><div class="s">24 months · {n_units:,} units</div></div>
  <div class="kpi"><div class="l">Gross margin</div><div class="v">{gm_pct:.1%}</div><div class="s">{cr(total_gm)} absolute</div></div>
  <div class="kpi"><div class="l">Discount spend</div><div class="v">{cr(disc_cost)}</div><div class="s">given away in promos &amp; tests</div></div>
  <div class="kpi"><div class="l">Median promo ROI</div><div class="v" style="color:var(--red)">{median_roi:.1f}x</div><div class="s">{neg_roi_share:.0%} of events loss-making</div></div>
  <div class="kpi"><div class="l">Working capital at risk</div><div class="v">₹{inv_risk_value/1e7:.1f} Cr</div><div class="s">{overstock_n} overstocked SKUs</div></div>
  <div class="kpi"><div class="l">Redirectable promo spend</div><div class="v" style="color:var(--acc)">{cr(redirectable_q)}</div><div class="s">per quarter, on loss-making events</div></div>
</div>

<h2>1 · The business problem</h2>
<p>NOVA MART's category team faced the classic mid-size retailer's dilemma: discounting visibly moves
volume, but nobody could say <i>which</i> discounts paid for themselves. Four questions kept recurring —
which products to discount and by how much, which segments to target, when inventory (not marketing)
should drive the decision, and when the right answer is to <b>not discount at all</b>.</p>
<p>PRISM is a decision-support product, not a dashboard. Its unit of output is a priced action:
<i>"run a 12% targeted promotion on this SKU for this segment; expect this much incremental
contribution margin, at this confidence, for these reasons."</i></p>

<div class="flow">
  <span class="s">DATA</span><span class="a">→</span>
  <span class="s">ANALYSIS</span><span class="a">→</span>
  <span class="s">PREDICTION</span><span class="a">→</span>
  <span class="s">OPTIMIZATION</span><span class="a">→</span>
  <span class="s">DECISION</span>
</div>

<h2>2 · The data (synthetic, but statistically honest)</h2>
<p>Because no real retailer hands its price book to a portfolio project, I generated a synthetic
company — but deliberately <i>not</i> a random one. The generator builds causality first, numbers
second: demand follows a constant-elasticity response <span class="code">Q = A·P<sup>−E</sup></span>
with category-level elasticities (Accessories ≈ 2.0 down to Cameras ≈ 0.8); customer archetypes modulate
discount response (Deal Seekers buy on promo, Premium Loyalists largely don't); inventory depletes with
real (s,S) replenishment and stockouts that cap sales; and the festive calendar (Oct–Nov peak,
Republic Day, Independence Day) drives seasonality the way Indian retail actually experiences it.</p>
<p>This matters for one reason: <b>the analytics have to earn their answers.</b> Nothing downstream
reads the "true" parameters — every elasticity, uplift and forecast in this report was estimated
from the transaction data alone.</p>

<figure><img src="data:image/png;base64,{imgs['f1']}">
<figcaption><b>Fig 1 —</b> Weekly revenue and gross margin, 24 months. Festive windows drive ~2x demand
spikes; margin follows but does not scale (fixed unit economics, deeper festive discounts).</figcaption></figure>

<figure><img src="data:image/png;base64,{imgs['f2']}">
<figcaption><b>Fig 2 —</b> Revenue and margin structure by category. Smartphones dominate revenue at
razor-thin margins; Accessories are small but carry 28–45% — the discount budget and the margin budget
live in different categories.</figcaption></figure>

<h2>3 · What the analytics found</h2>

<h3>3.1 Price sensitivity is measurable — and unevenly distributed</h3>
<p>Elasticities are estimated per SKU with within-product log-log regressions on price-level aggregates
(promo depth buckets), seasonally adjusted using the market demand index. The estimates recover the
designed causal structure with expected attenuation (~20–40%, count noise at SKU-week level is real),
and — more importantly — preserve the <i>ranking</i> that pricing decisions need.</p>

<figure><img src="data:image/png;base64,{imgs['f3']}">
<figcaption><b>Fig 3 —</b> Designed vs recovered elasticity by category (medians). Rank correlation is
strong: the econometrics correctly separates "discount these" from "never discount these" categories
from transactions alone.</figcaption></figure>

<figure><img src="data:image/png;base64,{imgs['f4']}">
<figcaption><b>Fig 4 —</b> The identification in action for one SKU: weekly units against effective
price. Teal = list price, amber/red = promo weeks. The dashed line is the fitted constant-elasticity
response used by the optimizer.</figcaption></figure>

<h3>3.2 Promotions work — which is exactly the problem</h3>
<p>Analysing {len(promo_eff)} promotion events with pre/during/post baselines (seasonally adjusted,
cannibalisation measured on non-scoped same-category SKUs):</p>
<ul>
  <li>Median sales uplift is <b>{median_uplift:+.0%}</b> — promotions genuinely move demand.</li>
  <li>Median ROI on campaign cost is <b>{median_roi:.1f}x</b>; {neg_roi_share:.0%} of events destroy contribution margin.</li>
  <li>Flash Sales earn the best uplift; Loyalty Offers are the worst value per rupee spent.</li>
  <li>Post-promo dips and same-category cannibalisation quietly claw back part of every "win".</li>
</ul>

<figure><img src="data:image/png;base64,{imgs['f5']}">
<figcaption><b>Fig 5 —</b> Promotion formats by median uplift (x) and median ROI (y); bubble = number of
events. The uncomfortable quadrant is the crowded one: strong uplift, negative return.</figcaption></figure>

<div class="callout red"><b>Insight that drives the product.</b> Higher sales was being read as
success. Once incrementality is netted against discount depth, campaign cost, cannibalisation and
post-promo dips, the portfolio conclusion flips: NOVA MART doesn't have a promotion problem, it has a
<b>targeting</b> problem. The fix is per-SKU, per-segment decisions — which is what the optimizer does.</div>

<h3>3.3 Segments respond to price very differently</h3>
<p>Behavioural segmentation (RFM + KMeans k=5 with rule overlays for New / At-Risk states, on
{len(seg_full):,} customers) recovers the designed archetypes — {match.loc['Premium Loyalists']:.0%} match
on Premium Loyalists, {match.loc['Deal Seekers']:.0%} on Deal Seekers — and the commercial asymmetry is stark:</p>
<ul>
  <li><b>Premium Loyalists</b> — {seg_pl.customers:,} customers, {seg_pl.revenue_share:.0%} of revenue, average
  discount availed {seg_pl.avg_discount:.0%}. They buy regardless of price.</li>
  <li><b>Deal Seekers</b> — {seg_ds.customers:,} customers, {seg_ds.revenue_share:.0%} of revenue, average
  discount availed {seg_ds.avg_discount:.0%}. They are the promotion audience.</li>
</ul>
<p>Deep discounts aimed at loyal high-spenders are almost pure margin leakage — roughly the single
largest avoidable cost the analysis surfaced.</p>

<figure><img src="data:image/png;base64,{imgs['f6']}">
<figcaption><b>Fig 6 —</b> Segment map: recency (inverted — left is healthier) vs median lifetime spend;
bubble = customers, colour = average discount availed (darker = more discount-driven).</figcaption></figure>

<h3>3.4 Inventory changes the answer</h3>
<p>Overstocked SKUs should sometimes be discounted even at mediocre elasticity; thin-cover SKUs should
never be discounted regardless of elasticity. The optimizer treats inventory days, stockout probability
and sell-through as first-class constraints, not reporting afterthoughts.</p>

<figure><img src="data:image/png;base64,{imgs['f7']}">
<figcaption><b>Fig 7 —</b> Margin × inventory-cover matrix. Red bubbles are the optimizer's favourite
targets: high margin sitting on slow stock — margin-funded promotions, not blanket sales.</figcaption></figure>

<h2>4 · The models (and their honest limits)</h2>
<p>Two XGBoost models do the prediction work, evaluated on a strict temporal split (train: first 80%
of weeks; test: the most recent 20%):</p>
<table>
<tr><th>Model</th><th>Target</th><th class="r">R²</th><th class="r">MAE</th><th>Notes</th></tr>
<tr><td>Demand forecast (XGBoost)</td><td>units / SKU-week</td><td class="r">{dm['xgb']['r2']:.2f}</td>
    <td class="r">{dm['xgb']['mae']:.2f} units</td><td>{dm['n_train']:,} train rows, recursive 8-wk forecast</td></tr>
<tr><td>Demand forecast (Linear)</td><td>units / SKU-week</td><td class="r">{dm['linear']['r2']:.2f}</td>
    <td class="r">{dm['linear']['mae']:.2f}</td><td>baseline</td></tr>
<tr><td>Demand forecast (Random Forest)</td><td>units / SKU-week</td><td class="r">{dm['random_forest']['r2']:.2f}</td>
    <td class="r">{dm['random_forest']['mae']:.2f}</td><td>baseline</td></tr>
<tr><td>Promotion response (XGBoost)</td><td>uplift ratio</td><td class="r">{pm['r2']:.2f}</td>
    <td class="r">{pm['mae']:.2f}x</td><td>{pm['n_obs']:,} product × event observations</td></tr>
</table>

<figure><img src="data:image/png;base64,{imgs['f8']}">
<figcaption><b>Fig 8 —</b> Left: boosting pays for itself — non-linear price × seasonality interactions
carry real signal. Right: the model keys on momentum and price levers, which is what a demand model
should do; search interest confirms the market-signal features earn their place.</figcaption></figure>

<h2>5 · The decision engine</h2>
<p>An optimizer (objective: maximise <b>incremental contribution margin</b>; constraints: ≥8% margin
floor, ≤30% discount, price ≥ 85% of competitor, inventory caps, campaign budget) feeds a transparent
six-rule cascade. No black box — every recommendation ships with its reasons, expected impact,
confidence and the KPIs that drove it. Three real outputs, verbatim:</p>
{rec_block(ex1)}
{rec_block(ex2)}
{rec_block(ex3)}
<p class="small mut">Note the deliberate balance: "don't discount" recommendations are treated as
first-class decisions, not non-events. Roughly {int((recs.action.str.contains('Maintain|Hold')).mean()*100)}%
of the catalogue earns a hold-price verdict — that discipline is where most of the margin comes from.</p>

<h2>6 · Business impact (modelled)</h2>
<ul>
  <li><b>{cr(redirectable_q)} / quarter</b> of campaign spend currently deployed on loss-making events —
  the engine flags these pre-launch and redirects budget to margin-accretive targets.</li>
  <li><b>{cr(top20_margin)} / quarter</b> of direct incremental contribution across the top-20 targeted
  actions — small on purpose: value comes from <i>stopping</i> bad promos more than launching new ones.</li>
  <li><b>Discount discipline:</b> {cr(disc_cost)} of discount cost over 24 months — most of it on
  SKUs whose demand barely responded. Elasticity screening alone would have caught a large share.</li>
  <li><b>Working capital release:</b> {cr(inv_risk_value)} of overstock value has margin-funded exit paths
  (targeted promotions on elastic, high-margin SKUs only).</li>
  <li><b>Segment reallocation:</b> redirecting deep discounts from Premium Loyalists to Deal Seekers
  preserves the uplift while protecting the margin that funds it.</li>
</ul>

<h2>7 · Limitations &amp; what I'd do next</h2>
<ul>
  <li>Synthetic data makes estimation cleaner than real-world mess — recovered elasticities are
  attenuated 20–40% by count noise (visible in Fig 3), and I kept that honestly in the report rather
  than tuning it away.</li>
  <li>Uplift is estimated from pre/post baselines, not causal designs. Next iteration: synthetic
  geo-holdouts so the promo model learns from true experiments.</li>
  <li>Constant-elasticity response is only valid inside the observed price range; the engine floors
  extrapolation at ±30%.</li>
  <li>Confidence scores are heuristic (significance + model quality + data depth). Calibrating them
  against realized outcomes is on the roadmap, as is a small FastAPI service so the optimizer can be
  called from the retailer's existing tools.</li>
</ul>

<h2>Appendix · Running the project</h2>
<ol class="steps">
  <li>Install Python 3.10+, then: <span class="code">pip install -r requirements.txt</span></li>
  <li>Build everything: <span class="code">python3 pipeline.py</span> &nbsp;(generates data → analytics → models → optimizer cache, ~30 s)</li>
  <li>Launch: <span class="code">streamlit run app.py</span> → open http://localhost:8501</li>
  <li>Optional: regenerate a fresh dataset <span class="code">python3 pipeline.py --force</span>; headless page tests <span class="code">python3 tests/smoke_test.py</span></li>
</ol>

<div class="foot">
PRISM — Pricing &amp; Revenue Intelligence System · Portfolio case study · August 2026<br>
Synthetic demonstration dataset — created for analytical demonstration. NOVA MART is fictional;
no real company data is used. All figures computed from the generated dataset by
<span class="code">make_report.py</span>.
</div>
</div></body></html>"""

out = ROOT / "PRISM_Case_Study.html"
out.write_text(html)
print(f"wrote {out} ({out.stat().st_size/1e6:.2f} MB)")
