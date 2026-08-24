"""PRISM synthetic data engine.

Generates a statistically-coherent 24-month retail dataset for NOVA MART.
Relationships are causal by construction:
  * demand responds to discounts via product-level price elasticity  Q ~ (1-d)^E
  * customer archetypes modulate discount response (Deal Seekers buy on promo)
  * inventory is simulated from actual sales with replenishment policy
  * seasonality follows the real Indian festive calendar (Oct–Nov peak)
  * competitor price gaps dampen/boost demand
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# --------------------------------------------------------------------------
def generate_customers(rng) -> pd.DataFrame:
    n = C.N_CUSTOMERS
    arch_names = list(C.SEGMENT_ARCHETYPES)
    arch_probs = [C.SEGMENT_ARCHETYPES[a]["share"] for a in arch_names]
    arch = rng.choice(arch_names, size=n, p=np.array(arch_probs) / sum(arch_probs))

    city_rows = rng.choice(len(C.CITIES), size=n, p=np.array([w for *_, w in C.CITIES]) /
                           sum(w for *_, w in C.CITIES))
    city = np.array([C.CITIES[i][0] for i in city_rows])
    state = np.array([C.CITIES[i][1] for i in city_rows])
    region = np.array([C.CITIES[i][2] for i in city_rows])

    age = np.clip(rng.normal(34, 11, n).astype(int), 18, 68)
    gender = rng.choice(["Male", "Female"], size=n, p=[0.62, 0.38])

    # income correlated with archetype
    inc_map = {"Low": 0, "Middle": 1, "Upper Middle": 2, "High": 3}
    income = np.empty(n, dtype=object)
    for a in arch_names:
        m = arch == a
        income[m] = rng.choice(C.SEGMENT_ARCHETYPES[a]["income"], size=m.sum())

    # occupation influenced by income & age
    occ = np.empty(n, dtype=object)
    for band in C.INCOME_BANDS:
        m = income == band
        if band == "High":
            p = [.28, .04, .18, .08, .14, .05, .02, .07, .04, .08, .02, 0]
        elif band == "Upper Middle":
            p = [.32, .10, .08, .07, .10, .06, .05, .08, .04, .05, .04, .01]
        elif band == "Middle":
            p = [.18, .22, .06, .04, .04, .12, .12, .09, .10, .02, .12, .01]
        else:
            p = [.05, .30, .03, .02, .01, .16, .14, .07, .12, .01, .20, .01]
        occ[m] = rng.choice(C.OCCUPATIONS, size=int(m.sum()), p=np.array(p) / np.sum(p))

    edu = np.empty(n, dtype=object)
    for band in C.INCOME_BANDS:
        m = income == band
        bias = inc_map[band]
        p = np.array([3 - bias * .4, 8 + bias, 48 + bias * 2, 26 + bias, 5 + bias * .8])
        edu[m] = rng.choice(C.EDUCATION, size=int(m.sum()), p=p / p.sum())

    marital = np.where(age > 30, rng.choice(["Married", "Single"], size=n, p=[.62, .38]),
                       rng.choice(["Married", "Single"], size=n, p=[.15, .85]))
    household = np.clip(rng.poisson(2.4, n) + (marital == "Married"), 1, 7)

    acq = rng.choice(C.ACQ_CHANNELS, size=n, p=[.24, .20, .16, .12, .16, .12])
    device = rng.choice(C.DEVICES, size=n, p=[.68, .26, .06])

    loyalty = np.empty(n, dtype=object)
    for a in arch_names:
        m = arch == a
        if a in ("Premium Loyalists",):
            loyalty[m] = rng.choice(C.LOYALTY_TIERS, size=int(m.sum()), p=[.05, .2, .45, .3])
        elif a in ("Deal Seekers", "Window Shoppers"):
            loyalty[m] = rng.choice(C.LOYALTY_TIERS, size=int(m.sum()), p=[.62, .28, .08, .02])
        else:
            loyalty[m] = rng.choice(C.LOYALTY_TIERS, size=int(m.sum()), p=[.35, .4, .2, .05])

    # signup dates: most pre-date the window; New Customers joined recently
    d0, d1 = pd.Timestamp(C.DATE_START), pd.Timestamp(C.DATE_END)
    signup = pd.Series(pd.Timestamp("2023-01-01") + pd.to_timedelta(
        rng.integers(0, (d0 - pd.Timestamp("2023-01-01")).days, n), unit="D"))
    new_m = arch == "New Customers"
    signup[new_m] = d1 - pd.to_timedelta(rng.integers(5, 150, new_m.sum()), unit="D")
    signup = pd.to_datetime(signup)

    # at-risk churn horizon (activity stops after churn_date)
    churn = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns]")
    risk_m = arch == "At-Risk High Value"
    churn[risk_m] = d1 - pd.to_timedelta(rng.integers(70, 240, risk_m.sum()), unit="D")

    df = pd.DataFrame(dict(
        customer_id=[f"CUS-{i:05d}" for i in range(n)],
        age=age, gender=gender, city=city, state=state, region=region,
        income_band=income, occupation=occ, education_level=edu,
        marital_status=marital, household_size=household, customer_segment=arch,
        acquisition_channel=acq, device_type=device, loyalty_tier=loyalty,
        signup_date=signup.dt.date))
    # latent simulation params (kept in generator, not exported)
    meta = dict(archetype=arch, churn_date=churn)
    return df, meta


# --------------------------------------------------------------------------
def _product_name(rng, cat, sub, brand, used: set) -> str:
    bank = C.NAME_BANK.get(sub, C.NAME_BANK.get(cat, (["Nova"], ["X"])))
    for _ in range(40):
        s = rng.choice(bank[0]); m = rng.choice(bank[1])
        name = f"{brand} {s} {m}".strip()
        if name not in used:
            used.add(name)
            return name
    name = f"{brand} {s} {m} {rng.integers(2, 99)}"
    used.add(name)
    return name


def generate_products(rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = C.N_PRODUCTS
    cats = list(C.CATEGORIES)
    per = n // len(cats)
    rows, extra = [], n - per * len(cats)
    for ci, cat in enumerate(cats):
        cnt = per + (1 if ci < extra else 0)
        for _ in range(cnt):
            rows.append(cat)
    cat_arr = np.array(rows)
    rng.shuffle(cat_arr)

    used: set = set()
    recs = []
    d1 = pd.Timestamp(C.DATE_END)
    for i, cat in enumerate(cat_arr):
        cc = C.CATEGORIES[cat]
        sub = rng.choice(cc["subs"])
        brand = rng.choice(C.BRANDS)
        lo, hi = cc["price"]
        # log-uniform price within category range
        price = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        price = round(price / 10) * 10
        margin = rng.uniform(*cc["margin"])
        cost = round(price * (1 - margin), -1)
        elas = rng.uniform(*cc["elas"])          # designed elasticity magnitude
        rating = float(np.clip(rng.normal(4.25, 0.42), 3.3, 4.9))
        lifecycle = rng.choice(C.LIFECYCLES, p=C.LIFECYCLE_W)
        if lifecycle == "Launch":
            intro = d1 - pd.to_timedelta(rng.integers(10, 100, 1)[0], unit="D")
        elif lifecycle == "Growth":
            intro = d1 - pd.to_timedelta(rng.integers(100, 320, 1)[0], unit="D")
        elif lifecycle == "Decline":
            intro = d1 - pd.to_timedelta(rng.integers(500, 1100, 1)[0], unit="D")
        else:
            intro = d1 - pd.to_timedelta(rng.integers(320, 900, 1)[0], unit="D")
        comp = price * np.exp(rng.normal(0, 0.075))
        comp = round(comp / 10) * 10
        seasonality = float(np.clip(rng.normal(cc["festive"] / 1.32, 0.15), 0.6, 1.6))
        recs.append(dict(
            product_id=f"PRD-{i:04d}",
            product_name=_product_name(rng, cat, sub, brand, used),
            category=cat, subcategory=sub, brand=brand,
            base_cost=cost, list_price=price,
            competitor_price=comp,
            rating=round(rating, 1),
            product_lifecycle=lifecycle,
            intro_date=intro,
            designed_elasticity=round(elas, 3),
            seasonality_index=round(seasonality, 2),
            supplier=rng.choice(C.SUPPLIERS),
            brand_popularity=float(np.clip(rng.normal(1, 0.15), 0.6, 1.5)),
        ))
    prods = pd.DataFrame(recs)
    # price tier within category (0 low, 1 mid, 2 high)
    prods["price_tier"] = prods.groupby("category")["list_price"].rank(pct=True).map(
        lambda p: 0 if p < .34 else (1 if p < .67 else 2)).astype(int)
    return prods


# --------------------------------------------------------------------------
def generate_promotions(rng, prods: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    d0, d1 = dates[0], dates[-1]
    recs = [("PRM-000", "No Discount", 0.0, d0, d1, "All", "All", 0, "None")]

    def add(pid, ptype, disc, s, e, tgt, cat, cost, chan):
        recs.append((pid, ptype, round(disc, 3), s, e, tgt, cat, cost, chan))

    pid = 1
    chans = ["Push & Email", "Social Media", "Homepage Banner", "Marketplace", "In-Store"]
    # festive tentpoles each year
    for year, (fest_s, fest_e, diw_s, diw_e) in {
            2024: ("2024-10-05", "2024-10-18", "2024-10-25", "2024-11-03"),
            2025: ("2025-10-05", "2025-10-18", "2025-10-12", "2025-10-21")}.items():
        add(f"PRM-{pid:03d}", "Seasonal Sale", rng.uniform(.10, .14), pd.Timestamp(fest_s),
            pd.Timestamp(fest_e), "All", "All", rng.integers(500_000, 900_000), rng.choice(chans)); pid += 1
        add(f"PRM-{pid:03d}", "Seasonal Sale", rng.uniform(.12, .16), pd.Timestamp(diw_s),
            pd.Timestamp(diw_e), "All", "All", rng.integers(600_000, 1_000_000), rng.choice(chans)); pid += 1
        add(f"PRM-{pid:03d}", "Seasonal Sale", rng.uniform(.09, .12), pd.Timestamp(f"{year+1}-01-20" if year == 2024 else "2026-01-20"),
            pd.Timestamp(f"{year+1}-01-28" if year == 2024 else "2026-01-28"), "All", "All",
            rng.integers(400_000, 700_000), rng.choice(chans)); pid += 1

    # monthly category flash sales
    cur = d0.to_period("M")
    while cur <= d1.to_period("M"):
        for _ in range(rng.integers(2, 4)):
            cat = rng.choice(list(C.CATEGORIES))
            s = pd.Timestamp(cur.to_timestamp()) + pd.Timedelta(days=int(rng.integers(0, 24)))
            e = s + pd.Timedelta(days=int(rng.integers(2, 5)))
            if e <= d1:
                add(f"PRM-{pid:03d}", "Flash Sale", rng.uniform(.12, .22), s, e, "All", cat,
                    int(rng.integers(120_000, 350_000)), rng.choice(chans)); pid += 1
        cur += 1

    # clearance events on overstock candidates
    slow = prods.sample(n=min(40, len(prods)), random_state=int(rng.integers(1e6)))
    for k in range(10):
        s = d0 + pd.Timedelta(days=int(rng.integers(20, (d1 - d0).days - 20)))
        e = s + pd.Timedelta(days=int(rng.integers(10, 15)))
        batch = slow.iloc[(k * 4) % len(slow):(k * 4) % len(slow) + 4]
        add(f"PRM-{pid:03d}", "Flat Discount", rng.uniform(.15, .25), s, e, "Deal Seekers",
            ";".join(batch.category.unique()), int(rng.integers(90_000, 240_000)), rng.choice(chans)); pid += 1

    # monthly loyalty offers
    cur = d0.to_period("M")
    while cur <= d1.to_period("M"):
        s = cur.to_timestamp() + pd.Timedelta(days=int(rng.integers(3, 12)))
        e = s + pd.Timedelta(days=int(rng.integers(4, 7)))
        if e <= d1:
            add(f"PRM-{pid:03d}", "Loyalty Offer", rng.uniform(.06, .09), s, e,
                "Premium Loyalists", "All", int(rng.integers(80_000, 160_000)), "Push & Email"); pid += 1
        cur += 1

    # bundles / BOGO on accessories
    acc = prods[prods.category.isin(["Accessories", "Headphones"])]
    cur = d0.to_period("M")
    while cur <= d1.to_period("M"):
        s = cur.to_timestamp() + pd.Timedelta(days=int(rng.integers(5, 20)))
        e = s + pd.Timedelta(days=int(rng.integers(5, 9)))
        if e <= d1:
            pick = acc.sample(n=min(3, len(acc)), random_state=int(rng.integers(1e6)))
            add(f"PRM-{pid:03d}", rng.choice(["Bundle", "Buy One Get One"]), rng.uniform(.15, .20),
                s, e, "All", ";".join(pick.category.unique()), int(rng.integers(60_000, 150_000)),
                rng.choice(chans)); pid += 1
        cur += 1

    promos = pd.DataFrame(recs, columns=["promotion_id", "promotion_type", "discount_pct",
                                         "start_date", "end_date", "target_segment", "category",
                                         "campaign_cost", "promotion_channel"])
    return promos


# --------------------------------------------------------------------------
def build_daily_discount_matrix(rng, prods, promos, dates):
    """(T x P) effective daily discount matrix + promo id matrix."""
    T, P = len(dates), len(prods)
    D = np.zeros((T, P))
    PRM = np.full((T, P), "PRM-000", dtype=object)
    day_idx = pd.Series(np.arange(T), index=dates)

    for _, pr in promos.iterrows():
        if pr.promotion_type == "No Discount":
            continue
        lo, hi = day_idx.get(pr.start_date, None), day_idx.get(pr.end_date, None)
        if lo is None or hi is None or lo > hi:
            continue
        # scope mask
        cats = str(pr.category).split(";")
        if "All" in cats:
            mask = np.ones(P, bool)
        else:
            mask = prods.category.isin(cats).values
        # loyalty offers only apply to premium brands rows in scope (approx via all)
        depth = np.clip(pr.discount_pct * rng.normal(1, 0.08, (1, mask.sum())), 0.3 * pr.discount_pct, 0.32)
        block = D[lo:hi + 1, mask]
        upd = np.maximum(block, depth)
        D[lo:hi + 1, mask] = upd
        PRM[lo:hi + 1, mask] = np.where(block < depth, pr.promotion_id, PRM[lo:hi + 1, mask])
    return D, PRM


def build_price_experiments(rng, prods, dates):
    """Standing ±3–9% price-test windows on a third of products (elasticity signal)."""
    T = len(dates)
    adj = np.ones((T, len(prods)))
    chosen = rng.choice(len(prods), size=100, replace=False)
    for p in chosen:
        for _w in range(int(rng.integers(2, 4))):        # 2-3 windows each
            s = rng.integers(0, T - 50)
            e = s + rng.integers(21, 42)
            adj[s:e, p] *= float(np.exp(rng.choice([-1, 1]) * rng.uniform(0.03, 0.09)))
    return adj


# --------------------------------------------------------------------------
def simulate(trans_path_seed, custs, cust_meta, prods, promos, dates, rng):
    T, P = len(dates), len(prods)
    D, PRM = build_daily_discount_matrix(rng, prods, promos, dates)
    ADJ = build_price_experiments(rng, prods, dates)

    listp = prods.list_price.values.astype(float)
    comp = prods.competitor_price.values.astype(float)
    price_t = ADJ * listp[None, :]                      # current price by day
    sell_price = price_t * (1 - D)                      # effective selling price

    # ---- demand multiplier matrix ----------------------------------------
    # constant-elasticity demand: Q = A * P^(-E).  A d% discount multiplies
    # quantity by (1-d)^(-E)  ->  e.g. d=15%, E=1.9  =>  1.40x units.
    elas = prods.designed_elasticity.values.astype(float)
    disc_mult = np.power(np.clip(1 - D, 0.6, 1), -elas[None, :])

    month = dates.month.values
    seas = np.ones((T, P))
    gaming = (prods.category.values == "Gaming")
    for m, v in C.MONTH_CURVE.items():
        seas[month == m, :] = v
    for m, v in C.MONTH_CURVE_GAMING.items():
        seas[np.ix_(month == m, gaming)] = v
    seas *= (prods.seasonality_index.values[None, :] / prods.seasonality_index.mean())
    seas *= np.where(dates.weekday.values[:, None] >= 5, 1.12, 1.0)   # weekends

    gap = sell_price / comp[None, :]
    comp_mult = np.clip(gap ** -0.55, 0.55, 1.5)

    # lifecycle ramp / decay
    intro_days = (dates[-1] - prods.intro_date).dt.days.values
    lifemul = np.ones(P)
    ramp = np.ones((T, P))
    for p in range(P):
        d_intro = (dates[-1] - prods.intro_date.iloc[p]).days
        start_idx = max(0, T - d_intro - 1) if d_intro < T else 0
        lc = prods.product_lifecycle.iloc[p]
        if lc == "Launch":
            ramp[start_idx:, p] = np.linspace(0.35, 1.15, T - start_idx)
        elif lc == "Growth":
            lifemul[p] = 1.12
        elif lc == "Decline":
            lifemul[p] = max(0.55, 1 - 0.16 * d_intro / 365)
    ramp *= lifemul[None, :]

    # weekly mean-reverting AR(1) demand noise per product (around 1.0)
    wn = np.ones((T, P))
    W = T // 7 + 1
    for p in range(P):
        walk = np.ones(W)
        shocks = rng.normal(0, 0.07, W)
        for w in range(1, W):
            walk[w] = 1 + 0.82 * (walk[w - 1] - 1) + shocks[w]
        wn[:, p] = np.clip(np.interp(np.arange(T), np.arange(W) * 7, walk), 0.55, 1.6)

    # market sentiment (shared, dips mid-2025)
    t_axis = np.arange(T)
    sent_series = 1 + 0.02 * np.sin(t_axis / 46)
    dip = np.exp(-0.5 * ((t_axis - T * 0.55) / (T * 0.09)) ** 2) * 0.10
    sent_series = np.clip(sent_series - dip, 0.85, 1.1)

    # ---- base velocity & calibration -------------------------------------
    med_cat = prods.groupby("category")["list_price"].transform("median").values
    vel_raw = ((med_cat / listp) ** 0.55
               * np.clip(prods.rating.values / 4.25, 0.82, 1.18) ** 1.2
               * prods.brand_popularity.values
               * np.exp(rng.normal(0, 0.18, P)))
    mult = disc_mult * seas * comp_mult * ramp * wn * sent_series[:, None]
    expected = (vel_raw[None, :] * mult).sum()
    TARGET_UNITS = 116_000
    scale = TARGET_UNITS / expected
    lam = vel_raw[None, :] * scale * mult
    units = rng.poisson(lam.astype(np.float64))

    print(f"[gen] expected units {expected*scale:,.0f}  sampled {units.sum():,}")

    # ---- assign customers per transaction ---------------------------------
    n = len(custs)
    arch = cust_meta["archetype"]
    churn = pd.to_datetime(cust_meta["churn_date"]).values
    signup = pd.to_datetime(custs.signup_date.values)
    valid_from = np.maximum(signup + np.timedelta64(6, "D"),
                            np.datetime64(dates[0]))
    valid_to = np.where(pd.isna(churn), np.datetime64(dates[-1]), churn)

    rate = np.array([C.SEGMENT_ARCHETYPES[a]["orders_per_yr"] for a in arch], float)
    deal_beta = np.array([C.SEGMENT_ARCHETYPES[a]["deal_beta"] for a in arch], float)
    low_pen = np.array([C.SEGMENT_ARCHETYPES[a]["low_disc_pen"] for a in arch], float)
    inc_idx = np.array([C.INCOME_BANDS.index(b) for b in custs.income_band])
    tier_match = np.array([[0.50, 1.00, 1.45], [0.72, 1.15, 1.18], [1.00, 1.10, 1.12],
                           [1.45, 1.02, 0.72]])[inc_idx]          # (customer x product-tier)

    affin = np.zeros((n, len(C.CATEGORIES)))
    for j, cat in enumerate(C.CATEGORIES):
        for a in C.SEGMENT_ARCHETYPES:
            m = arch == a
            affin[m, j] = 1.0 if cat in C.SEGMENT_ARCHETYPES[a]["cats"] else 0.22
    cat_idx = {c: i for i, c in enumerate(C.CATEGORIES)}

    base_w = rate[:, None] * np.where(affin > 0.5, affin, affin * 0.25)  # dampen non-affinity
    buckets = [(0.0, 0.03), (0.03, 0.10), (0.10, 0.20), (0.20, 0.35)]
    cum_bank = {}
    for ci in range(len(C.CATEGORIES)):
        w0 = base_w[:, ci]
        for bi, (blo, bhi) in enumerate(buckets):
            mid = (blo + bhi) / 2
            factor = low_pen if bi == 0 else 1 + deal_beta * mid
            factor = np.clip(factor, 0.05, 3.0)
            for tier in range(3):
                w = np.clip(w0 * factor * tier_match[:, tier], 1e-9, None)
                cum_bank[(ci, bi, tier)] = np.cumsum(w)

    dt64 = dates.values
    cust_ids = custs.customer_id.values
    cust_city = custs.city.values
    loyalty = custs.loyalty_tier.values

    pay_w = {"Bronze": [.52, .08, .16, .16, .04, .04], "Silver": [.46, .14, .16, .11, .05, .08],
             "Gold": [.40, .24, .15, .08, .06, .07], "Platinum": [.30, .42, .13, .04, .05, .06]}
    store_w = np.array([0.16, 0.2, 0.28, 0.34])[inc_idx]     # by income band

    # transactions, exploded per unit of demand
    promo_type_of = promos.set_index("promotion_id")["promotion_type"].to_dict()
    pid_arr = prods.product_id.values
    cost_arr = prods.base_cost.values.astype(float)
    cat_ret_arr = np.array([C.CATEGORIES[c]["ret"] for c in prods.category])

    rows = []
    tid = 0
    nonz = np.argwhere(units > 0)
    day_pos, prod_pos = nonz[:, 0], nonz[:, 1]
    order = np.argsort(day_pos, kind="stable")
    day_pos, prod_pos = day_pos[order], prod_pos[order]

    n_draw = len(day_pos)
    u_buf = rng.random(n_draw * 4 + 100)
    u_i = 0
    for k in range(n_draw):
        t, p = day_pos[k], prod_pos[k]
        qty_dem = int(units[t, p])
        if qty_dem == 0:
            continue
        d = D[t, p]
        ci = cat_idx[prods.category.iloc[p]]
        bi = 0 if d < 0.03 else (1 if d < 0.10 else (2 if d < 0.20 else 3))
        tier = int(prods.price_tier.iloc[p])
        cum = cum_bank[(ci, bi, tier)]
        total = cum[-1]
        dv = dt64[t]
        sp = float(sell_price[t, p])
        promo_id = PRM[t, p]
        is_flash = "Flash" in promo_type_of.get(promo_id, "")
        cost1 = cost_arr[p]
        pid1 = pid_arr[p]
        for _q in range(qty_dem):
            for _try in range(6):
                uu = u_buf[u_i % len(u_buf)]; u_i += 1
                cidx = int(np.searchsorted(cum, uu * total))
                cidx = min(cidx, n - 1)
                if valid_from[cidx] <= dv <= valid_to[cidx]:
                    break
            else:
                continue
            ret_p = cat_ret_arr[p]
            if arch[cidx] == "Deal Seekers":
                ret_p *= 1.35
            if is_flash:
                ret_p *= 1.25
            if d > 0.15:
                ret_p *= 1.10
            ret = 1 if rng.random() < ret_p else 0
            qty = 1
            if ci == cat_idx["Accessories"] and rng.random() < 0.22:
                qty = 2 + (rng.random() < 0.25)
            elif rng.random() < 0.04:
                qty = 2
            tid += 1
            rows.append((
                f"TXN-{tid:07d}", cust_ids[cidx], pid1, dv, qty, round(sp, 0),
                round(float(d), 4), promo_id,
                rng.choice(C.PAYMENTS, p=pay_w[loyalty[cidx]]),
                "Store" if rng.random() < store_w[cidx] else ("App" if rng.random() < 0.75 else "Website"),
                cust_city[cidx], ret, round(qty * sp, 2), round(qty * cost1, 2), 0.0))
    tx = pd.DataFrame(rows, columns=["transaction_id", "customer_id", "product_id",
                                     "transaction_date", "quantity", "selling_price",
                                     "discount_pct", "promotion_id", "payment_method",
                                     "channel", "city", "return_flag", "revenue",
                                     "cost", "gross_margin"])
    tx["gross_margin"] = tx.revenue - tx.cost
    return tx, D, PRM, ADJ, price_t


# --------------------------------------------------------------------------
def simulate_inventory(rng, prods, tx, dates):
    """Daily inventory ledger with (S, s)-style replenishment & stockouts."""
    P = len(prods)
    pid_idx = {p: i for i, p in enumerate(prods.product_id)}
    tx_ = tx.copy()
    tx_["day"] = tx_.transaction_date.map({d: i for i, d in enumerate(dates)})
    daily_sold = np.zeros((len(dates), P))
    for (d, p), q in tx_.groupby(["day", "product_id"])["quantity"].sum().items():
        daily_sold[int(d), pid_idx[p]] = q
    # returns arrive ~14 days later
    tx_ret = tx_[tx_.return_flag == 1]
    daily_ret = np.zeros((len(dates), P))
    for (d, p), q in tx_ret.groupby(["day", "product_id"])["quantity"].sum().items():
        dd = int(d) + 14
        if dd < len(dates):
            daily_ret[dd, pid_idx[p]] += q

    # policy classes: ~14% overstocked, ~9% understocked, rest normal
    pol = rng.choice(["normal", "over", "under"], size=P, p=[0.77, 0.14, 0.09])
    vel = np.zeros_like(daily_sold)
    csum = np.cumsum(np.vstack([np.zeros(P), daily_sold]), axis=0)
    for t in range(len(dates)):
        lo = max(0, t - 28)
        vel[t] = (csum[t + 1] - csum[lo]) / max(1, t - lo)
    vel = np.maximum(vel, 0.02)

    target_d = np.where(pol == "over", rng.uniform(110, 150, P),
               np.where(pol == "under", rng.uniform(13, 18, P), rng.uniform(38, 58, P)))
    rp_days = target_d * 0.55
    # initialise stock from full-horizon mean daily demand (opening-stock plan)
    mean_daily = np.maximum(daily_sold.mean(axis=0), 0.02)
    closing = (target_d * mean_daily * rng.uniform(0.9, 1.1, P)).astype(int)
    opening = closing.copy()
    on_order = np.zeros(P, dtype=int)
    lead = rng.integers(5, 9, P)
    arr_day = np.full(P, -1)
    inv_rows = []
    D = len(dates)
    for t in range(D):
        arrivals = arr_day == t
        closing = closing + np.where(arrivals, on_order, 0)
        on_order = np.where(arrivals, 0, on_order)
        arr_day[arrivals] = -1
        opening = closing.copy()
        sold = np.minimum(daily_sold[t], opening)
        lost = daily_sold[t] - sold
        closing = opening - sold + daily_ret[t]
        # Monday ordering
        if dates[t].weekday() == 0:
            proj = closing / vel[t]
            need = (proj < rp_days) & (on_order == 0)
            if need.any():
                qty = np.maximum((target_d * vel[t] - closing - on_order), 0).astype(int)
                on_order = np.where(need, qty, on_order)
                new_arr = t + lead
                arr_day = np.where(need & (arr_day < 0), new_arr, arr_day)
        inv_days = closing / np.maximum(vel[t], 0.02)
        inv_rows.append(pd.DataFrame(dict(
            date=dates[t], product_id=prods.product_id.values,
            opening_inventory=opening, units_sold=sold,
            units_returned=daily_ret[t].astype(int), closing_inventory=closing,
            inventory_days=np.round(inv_days, 1),
            stockout_flag=((closing <= 0) | (lost > 0)).astype(int),
            overstock_flag=(inv_days > 90).astype(int))))
    inv = pd.concat(inv_rows, ignore_index=True)
    return inv


# --------------------------------------------------------------------------
def build_market_signals(rng, prods, dates):
    rows = []
    t_axis = np.arange(len(dates))
    # economic sentiment: gentle walk with mid-2025 dip
    sent = 100 + 3 * np.sin(t_axis / 60) - 12 * np.exp(-0.5 * ((t_axis - len(dates) * 0.55) / (len(dates) * 0.10)) ** 2)
    sent = sent + rng.normal(0, 0.7, len(dates))
    month = dates.month.values
    for cat, cc in C.CATEGORIES.items():
        curve = np.array([(C.MONTH_CURVE_GAMING if cat == "Gaming" else C.MONTH_CURVE)[m] for m in month])
        comp_base = prods[prods.category == cat].competitor_price.mean()
        walk = np.cumsum(rng.normal(0, 0.0012, len(dates)))
        comp_price = comp_base * (1 - 0.07 * (curve - 1) / 0.6) * np.exp(walk)
        demand = 100 * (curve ** 0.9) * (sent / 100)
        # search interest LEADS demand by ~5 days
        lead_curve = np.roll(curve, -5); lead_curve[-5:] = curve[-5:]
        search = 100 * lead_curve * (sent / 100) * np.exp(rng.normal(0, 0.03, len(dates)))
        rows.append(pd.DataFrame(dict(
            date=dates, category=cat,
            competitor_avg_price=comp_price.round(1),
            market_demand_index=(demand * np.exp(rng.normal(0, 0.025, len(dates)))).round(1),
            seasonality_index=(100 * curve).round(1),
            search_interest_index=np.clip(search, 5, None).round(1),
            economic_sentiment_index=sent.round(1))))
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------
def run(seed=C.RANDOM_SEED, out=DATA_DIR):
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(C.DATE_START, C.DATE_END, freq="D")

    print("[gen] customers …")
    custs, cust_meta = generate_customers(rng)
    print("[gen] products …")
    prods = generate_products(rng)
    print("[gen] promotions …")
    promos = generate_promotions(rng, prods, dates)
    print("[gen] simulating 24 months of demand …")
    tx, D, PRM, ADJ, price_t = simulate(seed, custs, cust_meta, prods, promos, dates, rng)
    print(f"[gen] transactions: {len(tx):,} rows")
    print("[gen] inventory ledger …")
    inv = simulate_inventory(rng, prods, tx, dates)
    print("[gen] market signals …")
    signals = build_market_signals(rng, prods, dates)

    # final product state
    last_inv = inv.sort_values("date").groupby("product_id").tail(1).set_index("product_id")
    last_price = pd.Series(price_t[-1], index=prods.product_id.values)
    prods["current_price"] = prods.product_id.map(last_price).round(0)
    prods["inventory_units"] = prods.product_id.map(last_inv.closing_inventory).astype(int)
    prods["inventory_days"] = prods.product_id.map(last_inv.inventory_days).round(1)

    # keep transactions within stock availability: cap rows at units actually sold
    tx["_cum"] = tx.groupby(["transaction_date", "product_id"]).cumcount()
    allowed = inv.groupby(["date", "product_id"])["units_sold"].sum().to_dict()
    tx["_cap"] = [allowed.get((d, p), 10**9) for d, p in zip(tx.transaction_date, tx.product_id)]
    before = len(tx)
    tx = tx[tx._cum < tx._cap].drop(columns=["_cum", "_cap"])
    print(f"[gen] stockout-adjusted transactions: {len(tx):,} (dropped {before - len(tx):,})")

    tx.drop(columns=["_cap"], errors="ignore").to_csv(out / "transactions.csv", index=False)
    custs.to_csv(out / "customers.csv", index=False)
    prods.drop(columns=["intro_date", "designed_elasticity", "brand_popularity", "price_tier"]) \
        .to_csv(out / "products.csv", index=False)
    promos.assign(start_date=promos.start_date.dt.date, end_date=promos.end_date.dt.date) \
        .to_csv(out / "promotions.csv", index=False)
    inv.to_csv(out / "inventory.csv", index=False)
    signals.to_csv(out / "market_signals.csv", index=False)

    meta = dict(
        designed_elasticity={r.product_id: float(r.designed_elasticity) for r in prods.itertuples()},
        category_elasticity={c: float(np.mean([r.designed_elasticity for r in prods.itertuples() if r.category == c])) for c in C.CATEGORIES},
        n_transactions=len(tx), n_units=int(tx.quantity.sum()),
        revenue_total=float(tx.revenue.sum()), seed=seed,
        generated_at=pd.Timestamp.now().isoformat())
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[gen] done. revenue ₹{tx.revenue.sum()/1e7:.1f} Cr over 24 months")
    return meta


if __name__ == "__main__":
    run()
