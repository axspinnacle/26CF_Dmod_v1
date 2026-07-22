"""
compare_state_cols.py
---------------------
Quick script to compare st_raw_x vs insstate data quality.
Determines which column has better coverage (fewer NaN / empty).

Usage:
    python scripts/compare_state_cols.py

Output:
    Prints comparison table to console.
"""

import pandas as pd
import numpy as np

TRAIN_PATH = "/Users/Mach/dev/aps/data/2026_Dmodel_data/train_combined.parquet"
SAMPLE_N   = 50_000   # rows to sample

print(f"Loading {SAMPLE_N:,} rows ...")
df = pd.read_parquet(TRAIN_PATH).head(SAMPLE_N)
print(f"  Dataset: {df.shape[0]:,} rows x {df.shape[1]} cols\n")

cols_to_check = ["st_raw_x", "insstate"]
present = [c for c in cols_to_check if c in df.columns]
missing = [c for c in cols_to_check if c not in df.columns]

if missing:
    print(f"  NOT FOUND in dataset: {missing}\n")

print(f"{'Column':<15} {'dtype':<10} {'NaN count':>10} {'NaN %':>8} {'Empty str':>10} {'N unique':>9} {'Sample values'}")
print("-" * 90)

rows = []
for col in present:
    s         = df[col]
    dtype     = str(s.dtype)
    n_nan     = int(s.isna().sum())
    nan_pct   = n_nan / len(s) * 100
    n_empty   = int((s == "").sum()) if s.dtype == object else 0
    n_unique  = int(s.nunique(dropna=True))
    samples   = s.dropna().unique().tolist()[:8]
    print(f"{col:<15} {dtype:<10} {n_nan:>10,} {nan_pct:>7.2f}% {n_empty:>10,} {n_unique:>9,}  {samples}")
    rows.append({"col": col, "n_nan": n_nan, "nan_pct": nan_pct,
                 "n_empty": n_empty, "n_unique": n_unique})

print()
if len(rows) == 2:
    r0, r1 = rows
    if r0["n_nan"] < r1["n_nan"]:
        winner = r0["col"]
        loser  = r1["col"]
    elif r1["n_nan"] < r0["n_nan"]:
        winner = r1["col"]
        loser  = r0["col"]
    else:
        # tie — pick the one with more unique values
        winner = r0["col"] if r0["n_unique"] >= r1["n_unique"] else r1["col"]
        loser  = r1["col"] if winner == r0["col"] else r0["col"]

    print(f"Recommendation: USE  '{winner}'  (fewer NaN)")
    print(f"                DROP '{loser}'  (more NaN or duplicate)")
    print()
    print(f"To apply: add '{loser}' to ACTUARIAL_LIAB with strategy 'drop'")
    print(f"          in code/create_level_mapping.py and code/encoding_strategies.py")
