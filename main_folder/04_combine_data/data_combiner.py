import os
import pandas as pd


class DataCombiner:
    def __init__(self, config):
        self.global_cfg = config["global"]
        self.cfg = config["combine_data"]
        self.base_dir = self.global_cfg["base_data_dir"]
        self.join_key = self.cfg["join_key"]

        self.df_fold = None
        self.df_dep = None
        self.df_joined = None

    def _path(self, key):
        return os.path.join(self.base_dir, self.cfg["files"][key])

    def load_data(self):
        fold_path = self._path("fold_parquet")
        self.df_fold = pd.read_parquet(fold_path)
        print(f"Fold data shape: {self.df_fold.shape}")

        dep_path = self._path("dep_factor_parquet")
        self.df_dep = pd.read_parquet(dep_path)
        print(f"Dep factor data shape: {self.df_dep.shape}")

    def join(self):
        self.df_joined = self.df_dep.merge(self.df_fold, on=self.join_key, how="left")

        print(f"Joined data shape: {self.df_joined.shape}")
        print(f"Original dep factor rows: {len(self.df_dep):,}")
        print(f"Fold data rows: {len(self.df_fold):,}")

        matched = self.df_joined["fold"].notna().sum()
        unmatched = self.df_joined["fold"].isna().sum()
        total = len(self.df_joined)
        matched_pct = 100 * matched / total if total else 0
        unmatched_pct = 100 * unmatched / total if total else 0

        print(f"Rows with fold match: {matched:,} ({matched_pct:.2f}%)")
        print(f"Rows without fold match: {unmatched:,} ({unmatched_pct:.2f}%)")

    def export(self):
        output_path = self._path("output")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df_joined.to_parquet(output_path, index=False)
        print(f"Saved joined data: {output_path}")
        return self.df_joined

    def run_all(self):
        self.load_data()
        self.join()
        return self.export()
