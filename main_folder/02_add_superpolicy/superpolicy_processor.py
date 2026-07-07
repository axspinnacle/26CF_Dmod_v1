import os
import sys
import polars as pl

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "shared"))
from io_utils import describe_path, resolve_polars_source


class SuperpolicyProcessor:
    def __init__(self, config):
        self.global_cfg = config["global"]
        self.cfg = config["add_superpolicy"]
        self.base_dir = self.global_cfg["base_data_dir"]
        self.df = None

    def _path(self, key):
        return os.path.join(self.base_dir, self.cfg["files"][key])

    def load_data(self):
        input_path = self._path("input")
        describe_path(input_path)
        source = resolve_polars_source(input_path)

        cols = self.cfg["input_columns"]
        self.df = pl.scan_parquet(source).select(cols).collect()
        print(f"Loaded {self.df.shape[0]:,} rows")

    def consolidate(self):
        max_iterations = self.cfg["max_iterations"]

        self.df = self.df.with_row_index("ID")
        self.df = self.df.with_columns(
            pl.col("ID").min().over("policyid").alias("superpolicy_id")
        )

        for iteration in range(max_iterations):
            iter_num = iteration + 1

            crossing_vins = (
                self.df.group_by("vin")
                .agg(pl.col("superpolicy_id").n_unique().alias("policy_count"))
                .filter(pl.col("policy_count") > 1)
            )

            crossing_count = crossing_vins.shape[0]
            print(f"Iteration {iter_num:2d}: {crossing_count:,} VINs cross policies", end="")

            if crossing_count == 0:
                print(" -> converged")
                break

            print()

            self.df = self.df.with_columns(
                pl.col("superpolicy_id").min().over("vin").alias("superpolicy_id")
            )
            self.df = self.df.with_columns(
                pl.col("superpolicy_id").min().over("policyid").alias("superpolicy_id")
            )

        print(f"Consolidation complete after {iter_num + 1} iterations")

    def export(self):
        output_path = self._path("output")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        output_df = self.df.sort("ID").select(["vin_date", "superpolicy_id"])
        output_df.write_parquet(output_path)

        print(f"Output saved: {output_path}")
        print(f"Output shape: {output_df.shape}")
        return output_df

    def run_all(self):
        self.load_data()
        self.consolidate()
        return self.export()
