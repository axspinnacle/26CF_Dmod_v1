"""
test_data_processing.py
-----------------------
Unit tests for the data_processing module.
Uses synthetic in-memory DataFrames so no actual data files are required.
"""

import os
import sys
import json
import tempfile
import unittest

import pandas as pd
import numpy as np

# Make the code/ directory importable regardless of where tests are run from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from data_processing import (
    load_config,
    load_master_data,
    load_aux_data,
    join_data,
    check_data_quality,
    save_combined_data,
    save_quality_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_master_df() -> pd.DataFrame:
    """Return a small synthetic master DataFrame."""
    return pd.DataFrame(
        {
            "vin_date": ["VIN001_2024-01", "VIN002_2024-01", "VIN003_2024-01"],
            "make": ["Toyota", "Honda", "Ford"],
            "msrp": [25000.0, 22000.0, 30000.0],
            "mileage": [15000, 20000, 12000],
        }
    )


def _make_aux_df(include_extra: bool = False) -> pd.DataFrame:
    """Return a small synthetic auxiliary DataFrame.

    Parameters
    ----------
    include_extra : bool
        When True, an extra key is added that does *not* exist in master,
        so we can test row-drop detection.
    """
    data = {
        "vin_date": ["VIN001_2024-01", "VIN002_2024-01"],
        "dep_factor": [0.85, 0.90],
        "fold": [1, 2],
    }
    if include_extra:
        data["vin_date"].append("VIN999_2024-01")
        data["dep_factor"].append(0.75)
        data["fold"].append(3)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Tests – load_config
# ---------------------------------------------------------------------------

class TestLoadConfig(unittest.TestCase):
    def test_returns_dict(self):
        cfg = {
            "global": {"model_year": 2026},
            "preprocess": {"joinkey": "vin_date"},
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(cfg, f)
            tmp_path = f.name
        try:
            result = load_config(tmp_path)
            self.assertIsInstance(result, dict)
            self.assertEqual(result["global"]["model_year"], 2026)
            self.assertEqual(result["preprocess"]["joinkey"], "vin_date")
        finally:
            os.unlink(tmp_path)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_config("/nonexistent/path/config.json")


# ---------------------------------------------------------------------------
# Tests – load_master_data / load_aux_data
# ---------------------------------------------------------------------------

class TestLoadData(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _write_parquet(self, df: pd.DataFrame, filename: str) -> str:
        path = os.path.join(self.tmpdir, filename)
        df.to_parquet(path, index=False)
        return path

    def test_load_master_returns_dataframe(self):
        df = _make_master_df()
        self._write_parquet(df, "master.parquet")
        result = load_master_data(self.tmpdir, "master.parquet")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)

    def test_load_aux_returns_dataframe(self):
        df = _make_aux_df()
        self._write_parquet(df, "aux.parquet")
        result = load_aux_data(self.tmpdir, "aux.parquet")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)

    def test_load_master_missing_file_raises(self):
        with self.assertRaises(Exception):
            load_master_data(self.tmpdir, "nonexistent.parquet")


# ---------------------------------------------------------------------------
# Tests – join_data
# ---------------------------------------------------------------------------

class TestJoinData(unittest.TestCase):
    def test_inner_join_row_counts(self):
        master = _make_master_df()       # 3 rows
        aux = _make_aux_df()             # 2 rows, matching VIN001 & VIN002
        merged, report = join_data(master, aux, "vin_date")

        self.assertEqual(report["combined_rows"], 2)
        self.assertEqual(report["dropped_from_master"], 1)   # VIN003 dropped
        self.assertEqual(report["dropped_from_aux"], 0)
        self.assertEqual(report["master_rows"], 3)
        self.assertEqual(report["aux_rows"], 2)

    def test_inner_join_columns_present(self):
        master = _make_master_df()
        aux = _make_aux_df()
        merged, _ = join_data(master, aux, "vin_date")
        for col in ["vin_date", "make", "msrp", "dep_factor", "fold"]:
            self.assertIn(col, merged.columns)

    def test_pct_dropped_calculation(self):
        master = _make_master_df()       # 3 rows, 1 will be dropped
        aux = _make_aux_df()             # 2 rows
        _, report = join_data(master, aux, "vin_date")
        expected_pct = round(1 / 3 * 100, 4)
        self.assertAlmostEqual(report["pct_dropped_master"], expected_pct, places=2)

    def test_no_rows_dropped_when_keys_match(self):
        master = _make_master_df().iloc[:2]   # only VIN001 & VIN002
        aux = _make_aux_df()
        merged, report = join_data(master, aux, "vin_date")
        self.assertEqual(report["dropped_from_master"], 0)
        self.assertEqual(report["dropped_from_aux"], 0)

    def test_drop_report_keys_only_counts(self):
        master = _make_master_df()
        aux = _make_aux_df(include_extra=True)   # 3 rows, VIN999 not in master
        _, report = join_data(master, aux, "vin_date")
        self.assertEqual(report["keys_only_in_master"], 1)  # VIN003
        self.assertEqual(report["keys_only_in_aux"], 1)     # VIN999


# ---------------------------------------------------------------------------
# Tests – check_data_quality
# ---------------------------------------------------------------------------

class TestCheckDataQuality(unittest.TestCase):
    def _make_df_with_issues(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "vin_date": ["VIN001", None, "VIN003"],
                "msrp": [25000.0, 0.0, np.nan],
                "mileage": [0, 0, 10000],
            }
        )

    def test_returns_dataframe(self):
        df = _make_master_df()
        report = check_data_quality(df)
        self.assertIsInstance(report, pd.DataFrame)

    def test_correct_columns_present(self):
        df = _make_master_df()
        report = check_data_quality(df)
        for col in ["column", "dtype", "total_rows", "nan_count", "nan_pct",
                    "zero_count", "zero_pct", "unique_values"]:
            self.assertIn(col, report.columns)

    def test_nan_counts(self):
        df = self._make_df_with_issues()
        report = check_data_quality(df)
        msrp_row = report[report["column"] == "msrp"].iloc[0]
        self.assertEqual(msrp_row["nan_count"], 1)

    def test_zero_counts_numeric(self):
        df = self._make_df_with_issues()
        report = check_data_quality(df)
        mileage_row = report[report["column"] == "mileage"].iloc[0]
        self.assertEqual(mileage_row["zero_count"], 2)

    def test_zero_count_none_for_string(self):
        df = self._make_df_with_issues()
        report = check_data_quality(df)
        vin_row = report[report["column"] == "vin_date"].iloc[0]
        # When None is inserted into a pandas DataFrame column it becomes np.nan;
        # both represent "not applicable" for non-numeric columns.
        self.assertTrue(vin_row["zero_count"] is None or pd.isna(vin_row["zero_count"]))

    def test_one_row_per_column(self):
        df = _make_master_df()
        report = check_data_quality(df)
        self.assertEqual(len(report), len(df.columns))


# ---------------------------------------------------------------------------
# Tests – save_combined_data / save_quality_report
# ---------------------------------------------------------------------------

class TestSaveFunctions(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_save_combined_data_creates_file(self):
        df = _make_master_df()
        path = save_combined_data(df, self.tmpdir, "combined.parquet")
        self.assertTrue(os.path.isfile(path))

    def test_save_combined_data_roundtrip(self):
        df = _make_master_df()
        path = save_combined_data(df, self.tmpdir, "combined.parquet")
        loaded = pd.read_parquet(path)
        pd.testing.assert_frame_equal(df, loaded)

    def test_save_quality_report_creates_csv(self):
        df = _make_master_df()
        report = check_data_quality(df)
        path = save_quality_report(report, self.tmpdir, "reports", "qr.csv")
        self.assertTrue(os.path.isfile(path))
        self.assertTrue(path.endswith(".csv"))

    def test_save_quality_report_creates_subdir(self):
        df = _make_master_df()
        report = check_data_quality(df)
        subdir = "nested/reports"
        save_quality_report(report, self.tmpdir, subdir, "qr.csv")
        self.assertTrue(os.path.isdir(os.path.join(self.tmpdir, subdir)))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
