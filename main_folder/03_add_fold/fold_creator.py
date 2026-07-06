import os
import random
import time
import pandas as pd
import numpy as np


class FoldCreator:
    def __init__(self, config):
        self.global_cfg = config["global"]
        self.cfg = config["add_fold"]
        self.base_dir = self.global_cfg["base_data_dir"]

        self.join_key = self.cfg["join_key"]
        self.superpolicy_col = self.cfg["superpolicy_col"]
        self.n_folds = self.cfg["n_folds"]
        self.n_simulations = self.cfg["n_simulations"]

        self.ee_columns = self.cfg["ee_columns_raw"] if self.cfg["use_raw_ee"] else self.cfg["ee_columns_imps"]
        self.incurred_columns = self.cfg["incurred_columns_raw"] if self.cfg["use_raw_incurred"] else self.cfg["incurred_columns_imps"]

        self.all_coverages = self.cfg["all_coverages"]
        self.analysis_coverages = self.cfg["analysis_coverages"]
        self.objective_coverages = self.cfg["objective_coverages"]
        self.pp_columns = [f"pp_{cov}" for cov in self.analysis_coverages]

        self.df = None
        self.overall_stats = None
        self.results_df = None
        self.best_seed = None

    def _path(self, key):
        return os.path.join(self.base_dir, self.cfg["files"][key])

    def load_and_merge_data(self):
        main_path = self._path("main_data")
        ee_cols = list(self.ee_columns.values())
        inc_cols = list(self.incurred_columns.values())
        columns_to_load = [self.join_key] + ee_cols + inc_cols

        df_main = pd.read_parquet(main_path, columns=columns_to_load)
        print(f"Main data shape: {df_main.shape}")

        sp_path = self._path("superpolicy")
        df_sp = pd.read_parquet(sp_path)
        print(f"Superpolicy data shape: {df_sp.shape}")

        self.df = pd.merge(df_main, df_sp, on=self.join_key, how="inner")
        print(f"Merged data shape: {self.df.shape}")

        n_dup = self.df.duplicated(subset=self.join_key).sum()
        if n_dup > 0:
            print(f"WARNING: {n_dup} duplicate {self.join_key} values found")

    def calculate_pure_premium(self):
        for cov in self.analysis_coverages:
            ee_col = self.ee_columns[cov]
            inc_col = self.incurred_columns[cov]
            pp_col = f"pp_{cov}"

            self.df[pp_col] = self.df[inc_col] / self.df[ee_col]
            self.df[pp_col] = self.df[pp_col].replace([np.inf, -np.inf], np.nan)

        for cov in self.analysis_coverages:
            pp_col = f"pp_{cov}"
            print(f"  {pp_col}: mean={self.df[pp_col].mean():.2f}, median={self.df[pp_col].median():.2f}")

    def calculate_overall_stats(self):
        stats = {}
        total_ee = 0

        for cov in self.objective_coverages:
            ee_col = self.ee_columns[cov]
            inc_col = self.incurred_columns[cov]

            total_ee_cov = self.df[ee_col].sum()
            total_inc_cov = self.df[inc_col].sum()

            stats[cov] = {
                "mean_pp": total_inc_cov / total_ee_cov if total_ee_cov > 0 else 0,
                "total_ee": total_ee_cov,
            }
            total_ee += total_ee_cov

        for cov in self.objective_coverages:
            stats[cov]["norm_ee"] = stats[cov]["total_ee"] / total_ee if total_ee > 0 else 0

        self.overall_stats = stats

        for cov in self.objective_coverages:
            print(f"  {cov.upper()}: PP={stats[cov]['mean_pp']:.2f}, Weight={stats[cov]['norm_ee']:.4f}")

    def _calculate_fold_pp(self, df, fold_num):
        fold_data = df[df["fold"] == fold_num]
        pp_values = {}
        for cov in self.objective_coverages:
            ee_col = self.ee_columns[cov]
            inc_col = self.incurred_columns[cov]
            total_ee = fold_data[ee_col].sum()
            total_inc = fold_data[inc_col].sum()
            pp_values[cov] = total_inc / total_ee if total_ee > 0 else 0
        return pp_values

    def _objective_function(self, fold_pp):
        objective = 0
        for cov in self.objective_coverages:
            diff = abs(fold_pp[cov] - self.overall_stats[cov]["mean_pp"])
            weight = self.overall_stats[cov]["norm_ee"]
            objective += diff * weight
        return objective

    def _calculate_total_objective(self, df):
        fold_objectives = []
        for fold_num in range(1, self.n_folds + 1):
            fold_pp = self._calculate_fold_pp(df, fold_num)
            obj = self._objective_function(fold_pp)
            fold_objectives.append(obj)
        return np.mean(fold_objectives), fold_objectives

    def assign_folds(self, df, seed):
        df = df.copy()
        random.seed(seed)

        unique_ids = df[self.superpolicy_col].unique().tolist()
        unique_ids_sorted = sorted(unique_ids)
        shuffled_ids = random.sample(unique_ids_sorted, len(unique_ids_sorted))

        fold_size = len(shuffled_ids) // self.n_folds
        id_to_fold = {}
        for i in range(self.n_folds):
            start = i * fold_size
            end = start + fold_size if i < self.n_folds - 1 else len(shuffled_ids)
            for sp_id in shuffled_ids[start:end]:
                id_to_fold[sp_id] = i + 1

        df["fold"] = df[self.superpolicy_col].map(id_to_fold)
        return df

    def run_simulations(self):
        print(f"Running {self.n_simulations} simulations ({self.n_folds} folds)")
        results = []
        start_time = time.time()

        for seed in range(1, self.n_simulations + 1):
            iter_start = time.time()
            df_temp = self.assign_folds(self.df, seed)
            avg_obj, fold_objs = self._calculate_total_objective(df_temp)
            iter_time = time.time() - iter_start

            result = {"seed": seed, "avg_objective": avg_obj, "time_seconds": iter_time}
            for i, obj in enumerate(fold_objs):
                result[f"fold_{i+1}_obj"] = obj
            results.append(result)

            if seed % 10 == 0:
                elapsed = time.time() - start_time
                best = min(results, key=lambda r: r["avg_objective"])
                print(f"  {seed}/{self.n_simulations} done ({elapsed:.1f}s), best so far: seed {best['seed']} = {best['avg_objective']:.4f}")

        total_time = time.time() - start_time
        print(f"Total simulation time: {total_time:.1f}s")

        self.results_df = pd.DataFrame(results)
        return self.results_df

    def get_best_seed(self):
        best_idx = self.results_df["avg_objective"].idxmin()
        self.best_seed = int(self.results_df.loc[best_idx, "seed"])
        best_obj = self.results_df.loc[best_idx, "avg_objective"]
        print(f"Best seed: {self.best_seed} (objective={best_obj:.4f})")
        return self.best_seed, best_obj

    def apply_best_seed(self):
        self.df = self.assign_folds(self.df, self.best_seed)
        print("Fold distribution:")
        print(self.df["fold"].value_counts().sort_index())

    def validate_folds(self):
        print("--- Record Counts ---")
        fold_counts = self.df["fold"].value_counts().sort_index()
        total = len(self.df)
        for fold, count in fold_counts.items():
            pct = 100 * count / total
            print(f"  Fold {fold}: {count:,} records ({pct:.1f}%)")

        print("--- Superpolicy Counts ---")
        sp_counts = self.df.groupby("fold")[self.superpolicy_col].nunique()
        total_sp = self.df[self.superpolicy_col].nunique()
        for fold, count in sp_counts.items():
            pct = 100 * count / total_sp
            print(f"  Fold {fold}: {count:,} superpolicies ({pct:.1f}%)")

        print("--- Objective Function by Fold ---")
        for fold_num in range(1, self.n_folds + 1):
            fold_pp = self._calculate_fold_pp(self.df, fold_num)
            obj = self._objective_function(fold_pp)
            print(f"  Fold {fold_num}: {obj:.4f}")

        avg_obj, _ = self._calculate_total_objective(self.df)
        print(f"Average Objective: {avg_obj:.4f}")

    def export(self):
        output_path = self._path("output")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        sim_path = self._path("simulation_results")
        if self.results_df is not None:
            self.results_df.to_csv(sim_path, index=False)
            print(f"Simulation results saved: {sim_path}")

        output_cols = [self.join_key, self.superpolicy_col, "fold"] + self.pp_columns
        output_cols = [c for c in output_cols if c in self.df.columns]
        df_output = self.df[output_cols].copy()

        df_output.to_parquet(output_path, index=False)
        print(f"Fold assignments saved: {output_path}")
        print(f"Shape: {df_output.shape}")
        return df_output

    def run_all(self):
        self.load_and_merge_data()
        self.calculate_pure_premium()
        self.calculate_overall_stats()
        self.run_simulations()
        self.get_best_seed()
        self.apply_best_seed()
        self.validate_folds()
        return self.export()
