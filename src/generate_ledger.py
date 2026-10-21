"""Generate synthetic ledger / chart-of-accounts mapping data (Problem B).

Simulates a Business Central -> NetSuite style migration: every old-system
account is mapped to a new-system account code, and balances are posted
through that mapping. Each entity migrates its own chart, so account codes
are unique per entity (the entity is a code segment). A share of mappings
are wrong (injected), so the detector's job is to rank the mappings a
reviewer should check first.

Two realistic mis-mapping patterns are injected, always within one entity's
chart (that is where mapping mistakes happen):
- **swap**: two accounts' new codes are exchanged (each balance lands in the
  other account). Half of swaps are within one account category — the hard,
  realistic case of confusing two similar accounts.
- **collision**: an account is mapped to a code that already has an owner, so
  two source accounts post into one target and the target balance is the sum.

Posted balances carry +/-0.15% noise so a correct mapping does not have a
degenerate exactly-1.00 balance ratio. The label ``is_mismapped`` is 1 for
any account whose mapped code is not its own correct code. All data is
synthetic.
"""
import numpy as np
import pandas as pd

# category -> (old code prefix, new code prefix, base monthly magnitude, names)
COA = {
    "Revenue": ("41", "401", 180_000, [
        "Product Revenue", "Service Revenue", "Subscription Revenue",
        "Professional Services Revenue", "Other Income", "Licensing Revenue"]),
    "COGS": ("51", "501", 95_000, [
        "Cost of Goods Sold", "Freight and Shipping", "Inventory Adjustments",
        "Purchase Discounts", "Direct Labor Cost", "Warranty Costs"]),
    "Operating Expense": ("61", "601", 42_000, [
        "Rent Expense", "Utilities", "Marketing and Advertising",
        "Travel and Entertainment", "Software Subscriptions",
        "Office Supplies", "Insurance Expense", "Professional Fees",
        "Bank Fees", "Depreciation Expense"]),
    "Payroll": ("71", "701", 120_000, [
        "Salaries and Wages", "Payroll Taxes", "Employee Benefits",
        "Contractor Payments", "Commissions Expense", "Bonuses"]),
    "Assets": ("11", "101", 260_000, [
        "Cash and Equivalents", "Accounts Receivable", "Inventory",
        "Prepaid Expenses", "Fixed Assets", "Short-term Investments"]),
    "Liabilities": ("21", "201", 150_000, [
        "Accounts Payable", "Accrued Liabilities", "Deferred Revenue",
        "Sales Tax Payable", "Notes Payable", "Accrued Payroll"]),
}
ENTITIES = ["US", "UK", "DE", "IN"]

# Token substitutions used when the new system renames an account during
# migration ("Product Revenue" -> "Product Income"). Renames are realistic,
# and they stop name similarity from being a perfect oracle for the label:
# a correct mapping can have a dissimilar name, so the detector has to weigh
# the balance signals too.
RENAMES = {"Revenue": "Income", "Salaries": "Payroll", "Wages": "Pay",
           "Payments": "Disbursements", "Freight": "Shipping",
           "Supplies": "Materials", "Insurance": "Coverage", "Rent": "Lease",
           "Bonuses": "Bonus", "Taxes": "Tax", "Fees": "Charges"}


def _renamed(name: str) -> str:
    for old, new in RENAMES.items():
        if old in name:
            return name.replace(old, new, 1)
    return name + " (Migrated)"


def _build_accounts(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    seq = 0
    for category, (old_pre, new_pre, base, names) in COA.items():
        for entity in ENTITIES:
            for j, name in enumerate(names):
                seq += 1
                rows.append({
                    "account_id": f"A-{seq:04d}",
                    "old_code": f"{old_pre}{j}0-{entity}",
                    "new_code": f"{new_pre}{j}0-{entity}",
                    "account_name": name,
                    "category": category,
                    "entity": entity,
                    # Each account has its own typical monthly balance level.
                    "base_balance": float(base * rng.uniform(0.4, 1.6)),
                })
    return pd.DataFrame(rows)


def generate_ledger(n_months: int = 12, seed: int = 42,
                    mismap_share: float = 0.12) -> pd.DataFrame:
    """Generate one row per old account with old/new balances and the label."""
    rng = np.random.default_rng(seed)
    accounts = _build_accounts(rng)
    n = len(accounts)

    # 12 months of balances in the old system (random walk per account).
    noise = rng.normal(0, 0.06, size=(n, n_months))
    drift = np.cumsum(rng.normal(0, 0.02, size=(n, n_months)), axis=1)
    history = accounts["base_balance"].to_numpy()[:, None] * (1.0 + noise + drift)
    old_current = history[:, -1]
    accounts["old_balance"] = np.round(old_current, 2)
    accounts["old_balance_prev"] = np.round(history[:, -2], 2)
    accounts["old_balance_mean"] = np.round(history.mean(axis=1), 2)
    accounts["old_balance_std"] = np.round(history.std(axis=1), 2)

    # ~30% of accounts are renamed in the new chart (mapping stays correct).
    rename_mask = rng.random(n) < 0.30
    accounts["new_account_name"] = [(_renamed(nm) if m else nm)
                                    for nm, m in zip(accounts["account_name"], rename_mask)]

    # The mapping: every account starts mapped to its own correct new code.
    mapped = accounts["new_code"].to_numpy().copy()
    is_mis = np.zeros(n, dtype=int)
    entity_arr = accounts["entity"].to_numpy()
    category_arr = accounts["category"].to_numpy()

    # Inject mis-mappings inside each entity's own chart.
    for entity in ENTITIES:
        pool = np.where(entity_arr == entity)[0]
        n_mis = int(round(mismap_share * len(pool)))
        order = rng.permutation(pool)
        n_swap_accounts = int(round(0.6 * n_mis)) // 2 * 2
        swap_idx = list(order[:n_swap_accounts])
        # Pairing accounts that are adjacent in (category, balance size)
        # models the realistic hard case: adjacent accounts with similar
        # balances get confused during mapping.
        base_arr = accounts["base_balance"].to_numpy()
        swap_idx.sort(key=lambda i: (category_arr[i], base_arr[i]))
        pairs = [(swap_idx[k], swap_idx[k + 1])
                 for k in range(0, len(swap_idx) - 1, 2)]
        for a, b in pairs:
            mapped[a], mapped[b] = mapped[b], mapped[a]
            is_mis[a] = is_mis[b] = 1
        for a in order[n_swap_accounts:n_mis]:
            targets = pool[pool != a]
            t = int(rng.choice(targets))
            mapped[a] = mapped[t]
            is_mis[a] = 1

    accounts["mapped_new_code"] = mapped
    accounts["is_mismapped"] = is_mis

    # Balances posted into each new code = sum of old balances mapped there,
    # with small posting noise.
    posted = pd.Series(old_current).groupby(mapped).sum()
    posted = posted * rng.uniform(0.9985, 1.0015, size=len(posted))
    accounts["new_balance"] = np.round(
        accounts["mapped_new_code"].map(posted).to_numpy(), 2)

    # Name/category of the account that actually owns the mapped code —
    # this is what a reviewer sees in the new chart of accounts.
    owner = accounts.drop_duplicates("new_code").set_index("new_code")
    accounts["mapped_account_name"] = accounts["mapped_new_code"].map(owner["new_account_name"])
    accounts["mapped_category"] = accounts["mapped_new_code"].map(owner["category"])
    return accounts.drop(columns=["base_balance"])


if __name__ == "__main__":
    out = generate_ledger()
    out.to_csv("data/synthetic/ledger.csv", index=False)
    print(out["is_mismapped"].value_counts())
    print(f"{len(out)} accounts, {int(out['is_mismapped'].sum())} mis-mapped")
