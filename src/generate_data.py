"""Generate synthetic parallel-run commission data (Problem A).

Each row is a deal with a commission computed by the old logic and by the new
system. A subset of deals differ, and each difference is injected with a known
cause so the classifier can be evaluated. A small share of differing deals
carry TWO overlapping causes (the brief's worst case): the primary cause is in
``cause`` and the secondary in ``cause_secondary``.

Causes:
- ``none``     old and new agree exactly.
- ``rounding`` cent-level difference between the two calculations.
- ``rule``     the new plan logic pays a different rate.
- ``timing``   the deal crosses a period boundary; the commission lands in
               the next period, so the new system shows 0 for this period.
- ``data``     the deal amount differs between systems, or the deal is
               missing from the new system entirely (amount_new is NaN).

All data is synthetic. Nothing here is real company data.
"""
import numpy as np
import pandas as pd

CAUSES = ["none", "rounding", "rule", "timing", "data"]
DIFF_CAUSES = ["rounding", "rule", "timing", "data"]  # causes of a real difference
PLANS = {"Standard": 0.05, "Tiered": 0.06, "Strategic": 0.04}


def generate(n_deals: int = 6000, seed: int = 42, dual_share: float = 0.05) -> pd.DataFrame:
    """Generate the synthetic deal table.

    Parameters
    ----------
    n_deals: number of deals.
    seed: RNG seed; the same seed reproduces the same table exactly.
    dual_share: share of *differing* deals that receive a second,
        overlapping cause (rule+rounding or data+rule).
    """
    rng = np.random.default_rng(seed)
    month = rng.integers(1, 13, n_deals)
    day = rng.integers(1, 29, n_deals)
    plan = rng.choice(list(PLANS), n_deals, p=[0.5, 0.3, 0.2])
    rate = np.array([PLANS[p] for p in plan])
    amount_old = np.round(rng.lognormal(mean=9.2, sigma=0.8, size=n_deals), 2)
    crosses_period = rng.random(n_deals) < 0.12

    old = np.round(amount_old * rate, 2)
    new = old.copy()
    amount_new = amount_old.copy()
    cause = np.array(["none"] * n_deals, dtype=object)
    cause_secondary = np.array([""] * n_deals, dtype=object)

    # Pick which deals differ and assign a primary cause.
    differs = rng.random(n_deals) < 0.18
    idx = np.where(differs)[0]
    weights = [0.35, 0.25, 0.20, 0.20]  # rounding, rule, timing, data
    assigned = rng.choice(["rounding", "rule", "timing", "data"], size=len(idx), p=weights)

    for i, c in zip(idx, assigned):
        if c == "rounding":
            new[i] = np.round(old[i] + rng.choice([-0.02, -0.01, 0.01, 0.02]), 2)
        elif c == "rule":
            # New plan logic pays one point less than the old logic.
            new[i] = np.round(amount_old[i] * (rate[i] - 0.01), 2)
        elif c == "timing":
            crosses_period[i] = True
            new[i] = 0.0  # commission lands in the next period
        elif c == "data":
            if rng.random() < 0.3:
                amount_new[i] = np.nan  # deal missing in new system
                new[i] = 0.0
            else:
                factor = rng.uniform(0.7, 1.3)
                amount_new[i] = np.round(amount_old[i] * factor, 2)
                new[i] = np.round(amount_new[i] * rate[i], 2)
        cause[i] = c

    # Overlapping dual causes (worst case from the brief). A rule error can
    # hide under cent-level rounding noise; a data error can coincide with a
    # rate change. Timing deals are excluded: when the commission lands in
    # the next period, a second cause is not observable this period.
    dual_candidates = [i for i in idx
                       if cause[i] == "rule" or (cause[i] == "data" and not np.isnan(amount_new[i]))]
    n_dual = int(round(dual_share * len(idx)))
    dual_idx = rng.choice(dual_candidates, size=min(n_dual, len(dual_candidates)),
                          replace=False) if dual_candidates else np.array([], dtype=int)
    dual_set = set(int(i) for i in np.atleast_1d(dual_idx))
    for i in dual_set:
        if cause[i] == "rule":
            new[i] = np.round(new[i] + rng.choice([-0.02, -0.01, 0.01, 0.02]), 2)
            cause_secondary[i] = "rounding"
        elif cause[i] == "data":
            # Amount changed AND the new logic also pays one point less.
            new[i] = np.round(amount_new[i] * (rate[i] - 0.01), 2)
            cause_secondary[i] = "rule"

    # A little label noise: a few single-cause diffs are tagged by a reviewer
    # as another cause. Dual-cause rows are exempt so the overlap evaluation
    # stays clean.
    noisy = rng.random(n_deals) < 0.02
    flip_to = rng.choice(["rounding", "rule", "timing", "data"], n_deals)
    mask = noisy & (cause != "none") & np.array([i not in dual_set for i in range(n_deals)])
    cause = np.where(mask, flip_to, cause)

    df = pd.DataFrame({
        "deal_id": [f"D-{i:05d}" for i in range(n_deals)],
        "close_month": month,
        "close_date": [f"2026-{m:02d}-{d:02d}" for m, d in zip(month, day)],
        "plan": plan,
        "amount_old": amount_old,
        "amount_new": amount_new,
        "old_commission": old,
        "new_commission": new,
        "crosses_period": crosses_period.astype(int),
        "cause": cause,
        "cause_secondary": cause_secondary,
    })
    return df.sort_values(["close_month", "deal_id"]).reset_index(drop=True)


if __name__ == "__main__":
    out = generate()
    out.to_csv("data/synthetic/deals.csv", index=False)
    print(out["cause"].value_counts())
    print("dual-cause rows:", int((out["cause_secondary"] != "").sum()))
