#!/usr/bin/env python3
"""
Compare SAS output (sas7bdat) with Python output (CSV) to validate conversion
"""

import pandas as pd
from sas7bdat import SAS7BDAT
import sys

def read_sas_file(sas_file):
    """Read SAS7BDAT file into pandas DataFrame"""
    print(f"Reading SAS file: {sas_file}")
    with SAS7BDAT(sas_file) as file:
        df_sas = file.to_data_frame()
    return df_sas

def compare_dataframes(df_sas, df_python):
    """Compare two dataframes and report differences"""
    
    print("\n" + "="*70)
    print("  COMPARISON REPORT")
    print("="*70 + "\n")
    
    # Check shape
    print("[Shape Check]")
    print(f"  SAS file:    {df_sas.shape[0]:,} rows × {df_sas.shape[1]} columns")
    print(f"  Python CSV:  {df_python.shape[0]:,} rows × {df_python.shape[1]} columns")
    
    if df_sas.shape == df_python.shape:
        print("  ✓ Shapes match")
    else:
        print("  ✗ SHAPES DIFFER!")
        return False
    
    # Check column names
    print("\n[Column Names Check]")
    sas_cols = set(df_sas.columns)
    python_cols = set(df_python.columns)
    
    print(f"  SAS columns:    {sorted(df_sas.columns)}")
    print(f"  Python columns: {sorted(df_python.columns)}")
    
    if sas_cols == python_cols:
        print("  ✓ Column names match")
    else:
        missing_in_python = sas_cols - python_cols
        extra_in_python = python_cols - sas_cols
        if missing_in_python:
            print(f"  ✗ Missing in Python: {missing_in_python}")
        if extra_in_python:
            print(f"  ✗ Extra in Python: {extra_in_python}")
        return False
    
    # Check column dtypes
    print("\n[Data Types Check]")
    all_match = True
    for col in sorted(df_sas.columns):
        sas_dtype = df_sas[col].dtype
        python_dtype = df_python[col].dtype
        match = "✓" if sas_dtype == python_dtype else "✗"
        print(f"  {match} {col:20s} | SAS: {str(sas_dtype):15s} | Python: {str(python_dtype):15s}")
        if sas_dtype != python_dtype:
            all_match = False
    
    if all_match:
        print("  ✓ All data types match")
    
    # Check for NaN/nulls
    print("\n[Null Values Check]")
    sas_nulls = df_sas.isnull().sum()
    python_nulls = df_python.isnull().sum()
    
    for col in sorted(df_sas.columns):
        sas_null_cnt = sas_nulls[col]
        python_null_cnt = python_nulls[col]
        match = "✓" if sas_null_cnt == python_null_cnt else "✗"
        print(f"  {match} {col:20s} | SAS: {sas_null_cnt:8,d} | Python: {python_null_cnt:8,d}")
    
    # Check data values (sample comparison)
    print("\n[Data Value Comparison]")
    
    # Sort both by ID for consistent comparison
    df_sas_sorted = df_sas.sort_values('ID').reset_index(drop=True)
    df_python_sorted = df_python.sort_values('ID').reset_index(drop=True)
    
    # Compare first few rows
    print("\n  First 5 rows comparison:")
    print("\n  SAS:")
    print(df_sas_sorted.head().to_string())
    print("\n  Python:")
    print(df_python_sorted.head().to_string())
    
    # Check if all values match (allowing for type differences)
    mismatches = 0
    for col in df_sas.columns:
        # Convert to string for comparison to avoid type issues
        sas_vals = df_sas_sorted[col].astype(str)
        python_vals = df_python_sorted[col].astype(str)
        
        differences = (sas_vals != python_vals).sum()
        if differences > 0:
            print(f"\n  ✗ {col}: {differences:,} mismatches")
            # Show first few mismatches
            mismatch_indices = (sas_vals != python_vals).idxmax()
            for idx in range(min(3, len(df_sas))):
                if sas_vals.iloc[idx] != python_vals.iloc[idx]:
                    print(f"      Row {idx}: SAS={sas_vals.iloc[idx]}, Python={python_vals.iloc[idx]}")
            mismatches += differences
    
    if mismatches == 0:
        print("\n  ✓ All data values match!")
    else:
        print(f"\n  ✗ Total mismatches: {mismatches:,}")
        return False
    
    # Superpolicy statistics
    print("\n[Superpolicy Statistics Check]")
    sas_stats = df_sas_sorted.groupby('superpolicy_id').agg({
        'ID': 'count',
        'pol': 'nunique',
        'VIN': 'nunique'
    }).rename(columns={'ID': 'record_count'}).reset_index()
    
    python_stats = df_python_sorted.groupby('superpolicy_id').agg({
        'ID': 'count',
        'pol': 'nunique',
        'VIN': 'nunique'
    }).rename(columns={'ID': 'record_count'}).reset_index()
    
    print(f"  SAS superpolicies:    {len(sas_stats):,}")
    print(f"  Python superpolicies: {len(python_stats):,}")
    
    if len(sas_stats) == len(python_stats):
        print("  ✓ Same number of superpolicies")
    else:
        print("  ✗ Different number of superpolicies")
        return False
    
    # Compare key statistics
    print(f"\n  SAS records per superpolicy:")
    print(f"    Min: {sas_stats['record_count'].min():,}, Max: {sas_stats['record_count'].max():,}, Mean: {sas_stats['record_count'].mean():.1f}")
    print(f"\n  Python records per superpolicy:")
    print(f"    Min: {python_stats['record_count'].min():,}, Max: {python_stats['record_count'].max():,}, Mean: {python_stats['record_count'].mean():.1f}")
    
    print("\n" + "="*70)
    print("  ✓ VALIDATION COMPLETE - Files are equivalent!")
    print("="*70 + "\n")
    
    return True

if __name__ == "__main__":
    SAS_FILE = "/Users/Mach/dev/aps/data/2025_CFX_superpolicy/raw_out/superpolicy_final.sas7bdat"
    CSV_FILE = "/Users/Mach/dev/aps/data/2025_CFX_superpolicy/raw_out/superpolicy_final_polars.csv"
    
    try:
        print("Starting comparison...")
        df_sas = read_sas_file(SAS_FILE)
        print(f"  ✓ Loaded {len(df_sas):,} rows from SAS file\n")
        
        print(f"Reading CSV file: {CSV_FILE}")
        df_python = pd.read_csv(CSV_FILE)
        print(f"  ✓ Loaded {len(df_python):,} rows from CSV file\n")
        
        success = compare_dataframes(df_sas, df_python)
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
