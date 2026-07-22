"""
data_processing.py
------------------
Functions for loading, joining, and quality-checking the master and auxiliary
car depreciation datasets.

Split-processing additions
--------------------------
The following functions support memory-efficient fold-based splitting:
  - optimize_dtypes          : downcast float64→float32, int64→int32
  - load_master_data_optimized: load master and immediately optimize dtypes
  - load_aux_data_by_folds   : load aux filtered to a specific list of folds
  - save_split_data          : save a split DataFrame with a named convention
  - get_memory_usage         : report memory footprint of a DataFrame
"""

import gc
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


# ---------------------------------------------------------------------------
# Split-Processing  (memory-efficient fold-based pipeline)
# ---------------------------------------------------------------------------

def get_memory_usage(df: pd.DataFrame) -> str:
    """Return a human-readable string of a DataFrame's memory footprint.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to measure.

    Returns
    -------
    str
        Memory usage string, e.g. ``"1.23 GB"``.
    """
    bytes_used = df.memory_usage(deep=True).sum()
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if bytes_used < 1024:
            return f"{bytes_used:.2f} {unit}"
        bytes_used /= 1024
    return f"{bytes_used:.2f} PB"


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory usage.

    Conversions applied in-place on a copy of *df*:
    - ``float64`` → ``float32``   (50 % smaller; sufficient for XGBoost)
    - ``int64``   → ``int32``     (50 % smaller)

    Non-numeric columns (strings, categories, datetimes, etc.) are left
    unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        New DataFrame with optimized dtypes.
    """
    df = df.copy()
    for col in df.columns:
        col_dtype = df[col].dtype
        if col_dtype == "float64":
            df[col] = df[col].astype("float32")
        elif col_dtype == "int64":
            df[col] = df[col].astype("int32")
    return df


def load_master_data_optimized(base_dir: str, filename: str) -> pd.DataFrame:
    """Load the master dataset and immediately optimize numeric dtypes.

    This is a drop-in replacement for :func:`load_master_data` that converts
    ``float64`` → ``float32`` and ``int64`` → ``int32`` right after loading,
    cutting memory usage by roughly 50 % before any join takes place.

    Parameters
    ----------
    base_dir : str
        Base directory where the data files reside.
    filename : str
        Name of the master data Parquet file.

    Returns
    -------
    pd.DataFrame
        Loaded and dtype-optimized master dataset.
    """
    path = os.path.join(base_dir, filename)
    df = pd.read_parquet(path)
    df = optimize_dtypes(df)
    return df


def load_aux_data_by_folds(
    base_dir: str,
    filename: str,
    fold_list: list,
    fold_col: str = "fold",
) -> pd.DataFrame:
    """Load the auxiliary dataset and keep only rows belonging to *fold_list*.

    The fold column is used to filter the data **before** any join, so only
    the relevant slice of the aux file is held in memory.  Numeric dtypes are
    also optimized to ``float32`` / ``int32`` automatically.

    Parameters
    ----------
    base_dir : str
        Base directory where the data files reside.
    filename : str
        Name of the auxiliary data Parquet file.
    fold_list : list of int
        Fold numbers to retain (e.g. ``[1, 2, 3, 4, 5]`` for the training
        split).
    fold_col : str, optional
        Name of the fold column in the aux dataset (default: ``"fold"``).

    Returns
    -------
    pd.DataFrame
        Filtered and dtype-optimized auxiliary dataset.

    Raises
    ------
    KeyError
        If *fold_col* is not found in the auxiliary file.
    ValueError
        If no rows match the requested *fold_list*.
    """
    path = os.path.join(base_dir, filename)
    df = pd.read_parquet(path)

    if fold_col not in df.columns:
        raise KeyError(
            f"Fold column '{fold_col}' not found in '{filename}'. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[df[fold_col].isin(fold_list)].reset_index(drop=True)

    if df.empty:
        raise ValueError(
            f"No rows found for folds {fold_list} in column '{fold_col}'."
        )

    df = optimize_dtypes(df)
    return df


def save_split_data(
    df: pd.DataFrame,
    base_dir: str,
    split_name: str,
    quality_report_dir: str = "data_quality_reports",
) -> tuple[str, str]:
    """Save a split DataFrame as Parquet and its quality report as CSV.

    Output file names are derived automatically from *split_name*:
    - Parquet : ``<base_dir>/<split_name>_combined.parquet``
    - CSV     : ``<base_dir>/<quality_report_dir>/quality_report_<split_name>.csv``

    Parameters
    ----------
    df : pd.DataFrame
        Combined (joined) split dataset.
    base_dir : str
        Root data directory.
    split_name : str
        Short label for the split, e.g. ``"train"``, ``"test"``,
        ``"holdout"``.
    quality_report_dir : str, optional
        Sub-directory (relative to *base_dir*) for quality report CSVs
        (default: ``"data_quality_reports"``).

    Returns
    -------
    parquet_path : str
        Full path to the saved Parquet file.
    report_path : str
        Full path to the saved quality-report CSV.
    """
    # ── Parquet ──────────────────────────────────────────────────────────────
    parquet_filename = f"{split_name}_combined.parquet"
    parquet_path = save_combined_data(df, base_dir, parquet_filename)

    # ── Quality report ────────────────────────────────────────────────────────
    quality_df = check_data_quality(df)
    report_filename = f"quality_report_{split_name}.csv"
    report_path = save_quality_report(
        quality_df, base_dir, quality_report_dir, report_filename
    )

    return parquet_path, report_path
