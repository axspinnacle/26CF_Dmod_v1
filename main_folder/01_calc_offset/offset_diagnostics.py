import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class OffsetDiagnostics:
    def check_missing_values(self, df, model_year=2026):
        features = {
            "ODOMETER": "cef_est_curr_mi_grp_imps",
            "CALC_VEH_AGE": "dml_year_imps",
            "STATE": "st_raw",
            "BSST": "BODY_STYLE_SEGMENT_BODY_TYPE",
        }

        total_rows = len(df)
        print(f"Total records: {total_rows:,}")
        print(f"{'Feature':<25} {'Zeros':>12} {'Null':>12}")

        for name, col in features.items():
            if col not in df.columns:
                continue
            zeros = (df[col] == 0).sum() if pd.api.types.is_numeric_dtype(df[col]) else 0
            nulls = df[col].isna().sum()
            print(f"{name:<25} {zeros:>12,} {nulls:>12,}")

    def plot_dep_factor_distribution(self, df):
        dep_above_1 = df[df["Dep_factor"] > 1]["Dep_factor"].dropna()
        print(f"Records with Dep_factor > 1: {len(dep_above_1):,}")
        if len(dep_above_1) > 0:
            plt.hist(dep_above_1, bins=50)
            plt.xlabel("Dep_factor")
            plt.ylabel("Count")
            plt.title(f"Dep_factor > 1 (n={len(dep_above_1):,})")
            plt.show()

    def plot_age_odometer_heatmap(self, df, age_col="CALC_VEH_AGE", odo_col="cef_est_curr_mi_grp_imps", dep_col="Dep_factor"):
        from scipy.stats import binned_statistic_2d

        mask = df[age_col].notna() & df[odo_col].notna() & df[dep_col].notna()
        age = pd.to_numeric(df.loc[mask, age_col], errors="coerce").astype(float)
        odo = pd.to_numeric(df.loc[mask, odo_col], errors="coerce").astype(float) / 1000
        dep = pd.to_numeric(df.loc[mask, dep_col], errors="coerce").astype(float)

        valid = age.notna() & odo.notna() & dep.notna()
        age, odo, dep = age[valid].values, odo[valid].values, dep[valid].values

        age_bins = np.arange(0, 32, 2)
        odo_bins = np.arange(0, 251, 10)

        stat, x_edge, y_edge, _ = binned_statistic_2d(age, odo, dep, statistic="max", bins=[age_bins, odo_bins])

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(stat.T, origin="lower", aspect="auto", cmap="RdYlGn_r",
                        extent=[age_bins[0], age_bins[-1], odo_bins[0], odo_bins[-1]],
                        vmin=0.2, vmax=max(1.0, dep.max()))
        ax.set_xlabel("Vehicle Age (years)")
        ax.set_ylabel("Odometer (thousands of miles)")
        ax.set_title(f"Mean {dep_col} by Age and Odometer")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.show()

    def find_suspicious_records(self, df, age_min=18, age_max=24, dep_threshold=1.2):
        suspicious = df[
            (df["CALC_VEH_AGE"] >= age_min) &
            (df["CALC_VEH_AGE"] <= age_max) &
            (df["Dep_factor"] > dep_threshold)
        ]
        print(f"Suspicious records (age {age_min}-{age_max}, Dep_factor > {dep_threshold}): {len(suspicious):,}")
        return suspicious

    def compare_before_after(self, df_before, df_after, col):
        print(f"{col} before: mean={df_before[col].mean():.4f}, nulls={df_before[col].isna().sum():,}")
        print(f"{col} after:  mean={df_after[col].mean():.4f}, nulls={df_after[col].isna().sum():,}")
