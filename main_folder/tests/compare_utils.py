import pandas as pd


def compare_dataframes(df_old, df_new, key_col=None, float_tol=1e-4):
    print(f"Old shape: {df_old.shape}")
    print(f"New shape: {df_new.shape}")

    if df_old.shape[0] != df_new.shape[0]:
        print(f"WARNING: row count mismatch ({df_old.shape[0]} vs {df_new.shape[0]})")

    old_cols = set(df_old.columns)
    new_cols = set(df_new.columns)

    only_old = old_cols - new_cols
    only_new = new_cols - old_cols
    common = old_cols & new_cols

    if only_old:
        print(f"Columns only in old: {only_old}")
    if only_new:
        print(f"Columns only in new: {only_new}")

    if key_col and key_col in common:
        df_old = df_old.sort_values(key_col).reset_index(drop=True)
        df_new = df_new.sort_values(key_col).reset_index(drop=True)

    print(f"\nComparing {len(common)} common columns:")
    all_match = True

    for col in sorted(common):
        if col == key_col:
            continue

        if pd.api.types.is_numeric_dtype(df_old[col]) and pd.api.types.is_numeric_dtype(df_new[col]):
            diff = (df_old[col] - df_new[col]).abs()
            max_diff = diff.max()
            match = max_diff < float_tol or pd.isna(max_diff)
            status = "OK" if match else "FAIL"
            print(f"  [{status}] {col}: max_diff={max_diff}")
        else:
            match = (df_old[col].astype(str) == df_new[col].astype(str)).all()
            status = "OK" if match else "FAIL"
            mismatches = (~(df_old[col].astype(str) == df_new[col].astype(str))).sum()
            print(f"  [{status}] {col}: {mismatches} mismatches")

        if not match:
            all_match = False

    print(f"\nOverall match: {all_match}")
    return all_match


def compare_value_counts(df_old, df_new, col):
    old_counts = df_old[col].value_counts().sort_index()
    new_counts = df_new[col].value_counts().sort_index()

    print(f"--- {col} value counts ---")
    print(f"{'Value':<15} {'Old':>12} {'New':>12}")
    all_vals = sorted(set(old_counts.index) | set(new_counts.index))
    for val in all_vals:
        old_c = old_counts.get(val, 0)
        new_c = new_counts.get(val, 0)
        flag = "" if old_c == new_c else "  <-- DIFF"
        print(f"{str(val):<15} {old_c:>12,} {new_c:>12,}{flag}")


def summarize_diff(df_old, df_new, col):
    diff = df_new[col] - df_old[col]
    print(f"{col} diff stats: mean={diff.mean():.6f}, max={diff.abs().max():.6f}, nonzero={((diff.abs() > 1e-9).sum())}")
