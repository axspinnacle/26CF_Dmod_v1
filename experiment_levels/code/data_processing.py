"""
data_processing.py
------------------
Functions for loading, joining, and quality-checking the master and auxiliary
car depreciation datasets.
"""

import os
import json
import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load and return the project configuration from a JSON file.

    Parameters
    ----------
    config_path : str
        Absolute or relative path to config.json.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_master_data(base_dir: str, filename: str) -> pd.DataFrame:
    """Load the master dataset from a Parquet file.

    Parameters
    ----------
    base_dir : str
        Base directory where the data files reside.
    filename : str
        Name of the master data Parquet file.

    Returns
    -------
    pd.DataFrame
        Loaded master dataset.
    """
    path = os.path.join(base_dir, filename)
    df = pd.read_parquet(path)
    return df


def load_aux_data(base_dir: str, filename: str) -> pd.DataFrame:
    """Load the auxiliary car-depreciation-factor dataset from a Parquet file.

    Parameters
    ----------
    base_dir : str
        Base directory where the data files reside.
    filename : str
        Name of the auxiliary data Parquet file.

    Returns
    -------
    pd.DataFrame
        Loaded auxiliary dataset.
    """
    path = os.path.join(base_dir, filename)
    df = pd.read_parquet(path)
    return df


# ---------------------------------------------------------------------------
# Joining
# ---------------------------------------------------------------------------

def join_data(
    master_df: pd.DataFrame,
    aux_df: pd.DataFrame,
    join_key: str,
    join_type: str = "inner",
) -> tuple[pd.DataFrame, dict]:
    """Join master and auxiliary datasets on the specified key.

    After an inner join, rows that exist in one dataset but not the other are
    considered "dropped".  A summary of those counts is returned alongside the
    merged DataFrame so callers can inspect data-loss.

    Parameters
    ----------
    master_df : pd.DataFrame
        Master dataset.
    aux_df : pd.DataFrame
        Auxiliary dataset.
    join_key : str
        Column name to join on.
    join_type : str, optional
        Pandas merge *how* argument (default: ``"inner"``).

    Returns
    -------
    merged_df : pd.DataFrame
        The combined dataset.
    drop_report : dict
        Dictionary containing:
        - ``master_rows``  – original row count of master_df
        - ``aux_rows``     – original row count of aux_df
        - ``combined_rows``– row count after join
        - ``dropped_from_master`` – rows in master not matched in aux
        - ``dropped_from_aux``   – rows in aux not matched in master
        - ``pct_dropped_master`` – percentage of master rows dropped
        - ``pct_dropped_aux``   – percentage of aux rows dropped
    """
    master_rows = len(master_df)
    aux_rows = len(aux_df)

    merged_df = pd.merge(master_df, aux_df, on=join_key, how=join_type)
    combined_rows = len(merged_df)

    # Keys present in each dataset
    master_keys = set(master_df[join_key].unique())
    aux_keys = set(aux_df[join_key].unique())

    dropped_from_master = master_rows - combined_rows
    dropped_from_aux = aux_rows - combined_rows

    drop_report = {
        "master_rows": master_rows,
        "aux_rows": aux_rows,
        "combined_rows": combined_rows,
        "dropped_from_master": dropped_from_master,
        "dropped_from_aux": dropped_from_aux,
        "pct_dropped_master": round(dropped_from_master / master_rows * 100, 4) if master_rows else 0,
        "pct_dropped_aux": round(dropped_from_aux / aux_rows * 100, 4) if aux_rows else 0,
        "keys_only_in_master": len(master_keys - aux_keys),
        "keys_only_in_aux": len(aux_keys - master_keys),
        "common_keys": len(master_keys & aux_keys),
    }

    return merged_df, drop_report


# ---------------------------------------------------------------------------
# Data Quality
# ---------------------------------------------------------------------------

def check_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a per-column data quality report covering NaN and zero counts.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to profile.

    Returns
    -------
    pd.DataFrame
        Quality report with one row per column and the following fields:
        - ``column``
        - ``dtype``
        - ``total_rows``
        - ``nan_count``
        - ``nan_pct``
        - ``zero_count``  (for numeric columns; NaN for non-numeric)
        - ``zero_pct``    (for numeric columns; NaN for non-numeric)
        - ``unique_values``
    """
    total_rows = len(df)
    records = []

    for col in df.columns:
        nan_count = int(df[col].isna().sum())
        nan_pct = round(nan_count / total_rows * 100, 4) if total_rows else 0

        if pd.api.types.is_numeric_dtype(df[col]):
            zero_count = int((df[col] == 0).sum())
            zero_pct = round(zero_count / total_rows * 100, 4) if total_rows else 0
        else:
            zero_count = None
            zero_pct = None

        records.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "total_rows": total_rows,
                "nan_count": nan_count,
                "nan_pct": nan_pct,
                "zero_count": zero_count,
                "zero_pct": zero_pct,
                "unique_values": df[col].nunique(dropna=False),
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_combined_data(df: pd.DataFrame, base_dir: str, filename: str) -> str:
    """Save the combined DataFrame as a Parquet file.

    Parameters
    ----------
    df : pd.DataFrame
        Combined dataset to persist.
    base_dir : str
        Base directory for the output file.
    filename : str
        Output file name (e.g. ``"master_aux_combined.parquet"``).

    Returns
    -------
    str
        Full path to the saved file.
    """
    output_path = os.path.join(base_dir, filename)
    df.to_parquet(output_path, index=False)
    return output_path


def save_quality_report(
    quality_df: pd.DataFrame,
    base_dir: str,
    report_dir: str,
    report_filename: str,
) -> str:
    """Save the quality report DataFrame as a CSV file.

    The report is written into ``<base_dir>/<report_dir>/``.  The sub-directory
    is created automatically if it does not exist.

    Parameters
    ----------
    quality_df : pd.DataFrame
        Quality report produced by :func:`check_data_quality`.
    base_dir : str
        Root data directory.
    report_dir : str
        Sub-directory name (relative to base_dir) for quality reports.
    report_filename : str
        Output file name (e.g. ``"quality_report.csv"``).

    Returns
    -------
    str
        Full path to the saved CSV file.
    """
    report_folder = os.path.join(base_dir, report_dir)
    os.makedirs(report_folder, exist_ok=True)
    output_path = os.path.join(report_folder, report_filename)
    quality_df.to_csv(output_path, index=False)
    return output_path
