class CombineDiagnostics:
    def check_join_quality(self, df_joined, join_key="vin_date"):
        total = len(df_joined)
        dup_keys = df_joined[join_key].duplicated().sum()
        print(f"Total rows: {total:,}")
        print(f"Duplicate {join_key} values: {dup_keys:,}")

        for col in df_joined.columns:
            nulls = df_joined[col].isna().sum()
            if nulls > 0:
                pct = 100 * nulls / total
                print(f"  {col}: {nulls:,} nulls ({pct:.2f}%)")

    def compare_shapes(self, df_dep, df_fold, df_joined):
        print(f"Dep factor shape:  {df_dep.shape}")
        print(f"Fold shape:        {df_fold.shape}")
        print(f"Joined shape:      {df_joined.shape}")

        if df_joined.shape[0] != df_dep.shape[0]:
            print("WARNING: joined row count differs from dep factor row count (fan-out in join)")

    def fold_distribution(self, df_joined):
        if "fold" in df_joined.columns:
            print(df_joined["fold"].value_counts(dropna=False).sort_index())
