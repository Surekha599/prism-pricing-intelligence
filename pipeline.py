"""PRISM pipeline: data -> analytics -> ML models -> optimization -> cache.

Run:  python3 pipeline.py            (idempotent; reuses CSVs if present)
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import pandas as pd

from src import analytics, config as C, models, optimizer, segmentation
from src.data_generator import run as gen_run

ROOT = Path(__file__).resolve().parent
DATA, CACHE = ROOT / "data", ROOT / "cache"


def main(force=False):
    t0 = time.time()
    if force or not (DATA / "transactions.csv").exists():
        print("=== 1/5 generating synthetic dataset ===")
        gen_run()
    else:
        print("=== 1/5 dataset found — reusing ===")

    print("=== 2/5 loading tables ===")
    tx = pd.read_csv(DATA / "transactions.csv", parse_dates=["transaction_date"])
    cust = pd.read_csv(DATA / "customers.csv", parse_dates=["signup_date"])
    prods = pd.read_csv(DATA / "products.csv")
    promos = pd.read_csv(DATA / "promotions.csv", parse_dates=["start_date", "end_date"])
    inv = pd.read_csv(DATA / "inventory.csv", parse_dates=["date"])
    signals = pd.read_csv(DATA / "market_signals.csv", parse_dates=["date"])

    print("=== 3/5 analytics ===")
    elastic, cat_elastic, panel = analytics.estimate_elasticities(tx, prods, signals)
    print(f"  elasticity: {elastic.elasticity.notna().sum()} products estimated")
    promo_eff = analytics.promotion_effectiveness(tx, prods, promos, signals)
    promo_types = analytics.promo_type_summary(promo_eff)
    inv_health = analytics.inventory_health(inv, prods)
    seg_full, seg_g = segmentation.build_segments(tx, cust)
    seg_summary = segmentation.segment_summary(seg_full, tx)
    seg_elastic = analytics.estimate_segment_elasticity(tx, seg_full, prods, signals)
    # designed elasticity join for validation table
    import json
    meta = json.loads((DATA / "meta.json").read_text())
    elastic["designed"] = elastic.product_id.map(meta["designed_elasticity"])

    print("=== 4/5 ML models ===")
    feats, cat_codes = models.build_weekly_features(panel, prods, signals, inv)
    demand_model, dmetrics, cut = models.train_demand_model(feats)
    print(f"  demand model  R2={dmetrics['xgb']['r2']:.3f}  MAE={dmetrics['xgb']['mae']:.2f}")
    promo_ds = models.build_promo_dataset(tx, prods, promos, elastic.set_index("product_id").elasticity)
    promo_model, pmetrics = models.train_promo_model(promo_ds)
    print(f"  promo model   R2={pmetrics['r2']:.3f}  MAE={pmetrics['mae']:.3f}  n={pmetrics['n_obs']}")
    forecast = models.forecast_product(demand_model, feats, prods, signals, cat_codes=cat_codes)
    models.save_all(demand_model, dmetrics, promo_model, pmetrics, cut)

    print("=== 5/5 optimization & decisioning ===")
    recs = optimizer.build_recommendations(prods, elastic, inv_health, forecast, promo_eff, promo_ds=promo_ds)
    opp = optimizer.opportunity_scores(prods, elastic, inv_health, forecast, promo_eff)
    ctab = segmentation.validate_against_designed(seg_full)

    CACHE.mkdir(exist_ok=True)
    core = dict(
        elastic=elastic, cat_elastic=cat_elastic, panel=panel,
        promo_eff=promo_eff, promo_types=promo_types,
        inv_health=inv_health, seg_summary=seg_summary, seg_elastic=seg_elastic,
        seg_full=seg_full, ctab=ctab, dmetrics=dmetrics, pmetrics=pmetrics,
        forecast=forecast, recs=recs, opp=opp, cat_codes=cat_codes, cut_week=cut,
        promo_ds=promo_ds,
    )
    with open(CACHE / "core.pkl", "wb") as f:
        pickle.dump(core, f)
    print(f"done in {time.time()-t0:.0f}s -> cache/core.pkl "
          f"({(CACHE/'core.pkl').stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    import sys
    main(force="--force" in sys.argv)
