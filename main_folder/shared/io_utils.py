"""
Shared I/O helpers for handling parquet inputs that may be either:
  - a single .parquet file, or
  - a directory containing multiple .parquet part-files (e.g. output from a
    distributed job like Spark/Dask, or manually split exports).

pandas' `pd.read_parquet()` already handles both cases natively (pyarrow
auto-discovers part files in a directory and skips hidden/metadata files),
so no wrapper is needed there.

polars' `pl.scan_parquet()` / `pl.read_parquet()` is not consistent across
versions when given a bare directory path, so we resolve it to an explicit
glob pattern before handing it off.
"""

import os
import glob


def describe_path(path):
    """
    Print a short diagnostic of what `path` resolves to: a single file,
    or a directory containing N .parquet files (recursively).
    Useful when moving the pipeline to a new machine, to confirm the
    config path is pointing at something valid before loading.
    """
    if os.path.isdir(path):
        parquet_files = glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True)
        print(f"  [path] '{path}' is a DIRECTORY containing {len(parquet_files)} .parquet file(s)")
        if len(parquet_files) == 0:
            print(f"  WARNING: no .parquet files found under '{path}'")
    elif os.path.isfile(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  [path] '{path}' is a FILE ({size_mb:.1f} MB)")
    else:
        print(f"  WARNING: '{path}' does not exist (checked as both file and directory)")


def resolve_polars_source(path):
    """
    Given a path that may be a single .parquet file or a directory of
    .parquet part-files, return a source string/pattern safe to pass to
    pl.scan_parquet() / pl.read_parquet().

    - If `path` is a file, returns it unchanged.
    - If `path` is a directory, returns a recursive glob pattern
      f"{path}/**/*.parquet" so polars picks up all part files, including
      any nested partition subfolders.
    """
    if os.path.isdir(path):
        return os.path.join(path, "**", "*.parquet")
    return path
