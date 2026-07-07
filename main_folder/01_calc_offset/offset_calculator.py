import os
import sys
import json
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "shared"))
from io_utils import describe_path


class OffsetCalculator:
    def __init__(self, config):
        self.global_cfg = config["global"]
        self.cfg = config["calc_offset"]
        self.base_dir = self.global_cfg["base_data_dir"]
        self.model_year = self.global_cfg["model_year"]
        self.df = None
        self.fill_counts = {}

    def _path(self, key):
        return os.path.join(self.base_dir, self.cfg["files"][key])

    def load_data(self):
        cols = self.cfg["input_columns"]
        car_path = self._path("car_parquet")
        describe_path(car_path)
        # pandas read_parquet natively supports a directory of part-files,
        # so no special handling needed beyond the diagnostic above.
        self.df = pd.read_parquet(car_path, columns=cols)

        count_before = len(self.df)
        self.df = self.df[self.df["dml_year_imps"] != -999]
        deleted = count_before - len(self.df)
        pct = 100 * deleted / count_before if count_before else 0
        print(f"Records deleted (dml_year_imps = -999): {deleted:,} ({pct:.2f}%)")

    def calculate_geo_density(self):
        geo_path = self._path("zip_geodensity")
        df_geo = pd.read_csv(geo_path)

        self.df["REC_CNT"] = 1
        zip_counts = self.df.groupby(["zip"]).agg({"REC_CNT": "count"}).reset_index()

        geo = df_geo[["zip", "geo_pop_density"]].copy()
        geo = geo.loc[geo["geo_pop_density"].notna()]
        geo = geo.drop_duplicates(subset=["zip"])

        geo = geo.merge(zip_counts, how="left")
        geo["REC_CNT"] = geo["REC_CNT"].fillna(0)

        geo.sort_values(by="geo_pop_density", inplace=True)
        geo["cumsum"] = geo["REC_CNT"].cumsum()
        total_records = geo["cumsum"].max()

        if total_records > 0:
            geo["geo_pop_density_ntile"] = round(geo["cumsum"] / total_records, 2) * 100
        else:
            geo["geo_pop_density_ntile"] = 50

        geo["geo_pop_density_ntile"] = geo["geo_pop_density_ntile"].clip(0, 100)
        geo = geo[["zip", "geo_pop_density_ntile"]]

        self.df = self.df.merge(geo, how="left")

        gpdn_default_state = self.df.groupby(["st_raw"]).agg({"geo_pop_density_ntile": "mean"}).reset_index()
        gpdn_default_state.rename(columns={"geo_pop_density_ntile": "geo_pop_density_ntile_state_default"}, inplace=True)
        gpdn_default_state["geo_pop_density_ntile_state_default"] = round(gpdn_default_state["geo_pop_density_ntile_state_default"], 0)

        self.df = self.df.merge(gpdn_default_state, how="left")

        missing_mask = self.df["geo_pop_density_ntile"].isna()
        self.df["POP_DENSITY_IMP_FLAG"] = missing_mask.astype(int)

        self.df["geo_pop_density_ntile"] = self.df["geo_pop_density_ntile"].fillna(
            self.df["geo_pop_density_ntile_state_default"]
        ).fillna(self.cfg["default_pop_density_percentile"])

        del self.df["geo_pop_density_ntile_state_default"]
        del self.df["REC_CNT"]

        out_of_range = ((self.df["geo_pop_density_ntile"] < 0) | (self.df["geo_pop_density_ntile"] > 100)).sum()
        if out_of_range > 0:
            raise ValueError(f"geo_pop_density_ntile has {out_of_range} values outside 0-100 range")

        pop_imputed = self.df["POP_DENSITY_IMP_FLAG"].sum()
        total = len(self.df)
        pct = 100 * pop_imputed / total if total else 0
        print(f"Pop density imputed: {pop_imputed:,} ({pct:.2f}%)")
        self.fill_counts["pop_density"] = int(pop_imputed)

    def merge_bsst(self):
        vin_bsst_path = self._path("vin_bsst")
        df_bsst = pd.read_parquet(vin_bsst_path, columns=self.cfg["vin_bsst_columns"])
        self.df = self.df.merge(df_bsst, left_on="vin", right_on="VIN", how="left")

        self.df["BSST_formatted"] = (
            self.df["BODY_STYLE_SEGMENT_BODY_TYPE"]
            .str.replace(" ", "_")
            .str.replace("_(", "_", regex=False)
            .str.replace(")", "", regex=False) + "_GLM"
        )

    def apply_odometer_defaults(self):
        odo_defaults = pd.read_csv(self._path("default_odometer"))
        max_age = self.cfg["max_age_for_odometer_lookup"]

        self.df["CALC_VEH_AGE"] = self.model_year - self.df["dml_year_imps"]
        age_for_lookup = self.df["CALC_VEH_AGE"].clip(lower=0, upper=max_age)

        missing_mask = (self.df["cef_est_curr_mi_grp_imps"] == 0) | self.df["cef_est_curr_mi_grp_imps"].isna()
        self.df["ODOMETER_IMP_FLAG"] = missing_mask.astype(int)

        odo_lookup = odo_defaults.set_index("CALC_VEH_AGE")["ODOMETER_mean"].to_dict()
        self.df.loc[missing_mask, "cef_est_curr_mi_grp_imps"] = age_for_lookup[missing_mask].map(odo_lookup)

        odo_imputed = missing_mask.sum()
        total = len(self.df)
        pct = 100 * odo_imputed / total if total else 0
        print(f"Odometer imputed: {odo_imputed:,} ({pct:.2f}%)")
        self.fill_counts["odometer"] = int(odo_imputed)

    def apply_state_defaults(self):
        state_defaults = pd.read_csv(self._path("default_state_by_bsst"))
        missing_mask = self.df["st_raw"].isna() | (self.df["st_raw"] == "")

        if missing_mask.sum() == 0:
            self.fill_counts["state"] = 0
            return

        state_lookup = state_defaults.set_index("BSST")["STATE_mode"].to_dict()
        self.df.loc[missing_mask, "st_raw"] = self.df.loc[missing_mask, "BODY_STYLE_SEGMENT_BODY_TYPE"].map(state_lookup)

        filled = missing_mask.sum() - (self.df["st_raw"].isna() | (self.df["st_raw"] == "")).sum()
        self.fill_counts["state"] = int(filled)

    def predict_dep_factor(self):
        from glm_predictor import load_glm_model, predict_glm

        bsst_glm_path = self._path("bsst_glm_folder")
        unique_bsst = self.df["BSST_formatted"].dropna().unique()

        self.df["Dep_factor"] = pd.NA
        debug_folder = self._path("debug_folder")
        threshold = self.cfg["dep_factor_warning_threshold"]

        for bsst in unique_bsst:
            model_path = os.path.join(bsst_glm_path, f"{bsst}.json")
            if not os.path.exists(model_path):
                continue

            mask = self.df["BSST_formatted"] == bsst
            if mask.sum() == 0:
                continue

            with open(model_path, "r") as f:
                glm_model = json.load(f)

            df_segment = self.df[mask].copy()
            df_segment["ODOMETER"] = df_segment["cef_est_curr_mi_grp_imps"]

            for state in df_segment["st_raw"].unique():
                if pd.notna(state):
                    df_segment[f"STATE_{state}"] = (df_segment["st_raw"] == state)

            for make in df_segment["dml_make_raw"].unique():
                if pd.notna(make):
                    df_segment[f"MAKE_{make}"] = (df_segment["dml_make_raw"] == make)

            predictions = predict_glm(df_segment, glm_model)
            df_segment["Dep_factor"] = predictions.values

            mask_problem = df_segment["Dep_factor"] > threshold
            if mask_problem.any():
                os.makedirs(debug_folder, exist_ok=True)
                debug_path = os.path.join(debug_folder, f"Records_{bsst}.parquet")
                df_segment[mask_problem].to_parquet(debug_path, index=False)

            self.df.loc[mask, "Dep_factor"] = predictions.values

        self.df["veh_value_dep"] = self.df["vc_msrp_impa"] * self.df["Dep_factor"]

    def export(self):
        out_cols = self.cfg["output_columns"]
        out_cols = [c for c in out_cols if c in self.df.columns]
        df_out = self.df[out_cols].copy()

        output_path = self._path("output")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_out.to_parquet(output_path, index=False)

        print(f"Exported {len(df_out):,} records to {output_path}")
        return df_out

    def run_all(self):
        self.load_data()
        self.merge_bsst()
        self.apply_odometer_defaults()
        self.apply_state_defaults()
        self.calculate_geo_density()
        self.predict_dep_factor()
        return self.export()
