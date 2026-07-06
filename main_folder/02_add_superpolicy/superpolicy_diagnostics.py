class SuperpolicyDiagnostics:
    def summary_stats(self, df):
        stats = {
            "Total Records": df.shape[0],
            "Original Policies": df["policyid"].n_unique(),
            "Consolidated Policies": df["superpolicy_id"].n_unique(),
            "Unique VINs": df["vin"].n_unique(),
        }
        orig = stats["Original Policies"]
        cons = stats["Consolidated Policies"]
        stats["Policy Reduction"] = f"{((orig - cons) / orig * 100):.2f}%"

        for key, value in stats.items():
            print(f"{key:25s}: {value:>20}")
        return stats

    def null_check(self, df):
        cols = ["ID", "policyid", "vin", "vin_date", "superpolicy_id"]
        for col in cols:
            if col in df.columns:
                count = df[col].null_count()
                status = "OK" if count == 0 else "FAIL"
                print(f"  {status} {col}: {count}")

    def policy_size_distribution(self, df):
        import polars as pl
        policy_sizes = df.group_by("superpolicy_id").agg(pl.len().alias("record_count"))

        print(f"Min:  {policy_sizes['record_count'].min():,}")
        print(f"Max:  {policy_sizes['record_count'].max():,}")
        print(f"Mean: {policy_sizes['record_count'].mean():.0f}")
        print(f"Median: {policy_sizes['record_count'].median():.0f}")
