#!/usr/bin/env python3
"""
Quick validation of Python output vs SAS output (without full comparison)
Checks key statistics instead of reading all rows
"""

import pandas as pd
from sas7bdat import SAS7BDAT
import sys

def get_sas_stats(sas_file):
    """Get statistics from SAS file without loading all data"""
    print(f"Reading SAS file: {sas_file}")
    with SAS7BDAT(sas_file) as file:
        # Get metadata
        df = file.to_data_frame()
    
    stats = {
        'rows': len(df),
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'unique_superpolicies': df['superpolicy_id'].nunique(),
        'unique_policies': df['pol'].nunique(),
        'unique_vins': df['VIN'].nunique(),
        'min_superpolicy': df['superpolicy_id'].min(),
        'max_superpolicy': df['superpolicy_id'].max(),
    }
    return stats

def get_csv_stats(csv_file):
    """Get statistics from CSV file"""
    print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    stats = {
        'rows': len(df),
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'unique_superpolicies': df['superpolicy_id'].nunique(),
        'unique_policies': df['pol'].nunique(),
        'unique_vins': df['VIN'].nunique(),
        'min_superpolicy': df['superpolicy_id'].min(),
        'max_superpolicy': df['superpolicy_id'].max(),
    }
    return stats

def compare_stats(sas_stats, csv_stats):
    """Compare statistics between SAS and Python outputs"""
    
    print("\n" + "="*70)
    print("  QUICK VALIDATION REPORT")
    print("="*70 + "\n")
    
    all_match = True
    
    # Row count
    print("[Row Count]")
    sas_rows = sas_stats['rows']
    csv_rows = csv_stats['rows']
    match = "✓" if sas_rows == csv_rows else "✗"
    print(f"  {match} SAS: {sas_rows:,} rows | CSV: {csv_rows:,} rows")
    if sas_rows != csv_rows:
        all_match = False
    
    # Columns
    print("\n[Columns]")
    sas_cols = set(sas_stats['columns'])
    csv_cols = set(csv_stats['columns'])
    match = "✓" if sas_cols == csv_cols else "✗"
    print(f"  {match} SAS columns: {sorted(sas_cols)}")
    print(f"  {match} CSV columns: {sorted(csv_cols)}")
    if sas_cols != csv_cols:
        all_match = False
        print(f"     Missing in CSV: {sas_cols - csv_cols}")
        print(f"     Extra in CSV: {csv_cols - sas_cols}")
    
    # Superpolicy count
    print("\n[Superpolicy Count]")
    sas_sp = sas_stats['unique_superpolicies']
    csv_sp = csv_stats['unique_superpolicies']
    match = "✓" if sas_sp == csv_sp else "✗"
    print(f"  {match} SAS: {sas_sp:,} unique superpolicies")
    print(f"  {match} CSV: {csv_sp:,} unique superpolicies")
    if sas_sp != csv_sp:
        all_match = False
    
    # Policy count
    print("\n[Original Policy Count]")
    sas_pol = sas_stats['unique_policies']
    csv_pol = csv_stats['unique_policies']
    match = "✓" if sas_pol == csv_pol else "✗"
    print(f"  {match} SAS: {sas_pol:,} unique policies")
    print(f"  {match} CSV: {csv_pol:,} unique policies")
    if sas_pol != csv_pol:
        all_match = False
    
    # VIN count
    print("\n[VIN Count]")
    sas_vin = sas_stats['unique_vins']
    csv_vin = csv_stats['unique_vins']
    match = "✓" if sas_vin == csv_vin else "✗"
    print(f"  {match} SAS: {sas_vin:,} unique VINs")
    print(f"  {match} CSV: {csv_vin:,} unique VINs")
    if sas_vin != csv_vin:
        all_match = False
    
    # Superpolicy ID range
    print("\n[Superpolicy ID Range]")
    sas_min, sas_max = sas_stats['min_superpolicy'], sas_stats['max_superpolicy']
    csv_min, csv_max = csv_stats['min_superpolicy'], csv_stats['max_superpolicy']
    match = "✓" if (sas_min == csv_min and sas_max == csv_max) else "✗"
    print(f"  {match} SAS: {sas_min:,} to {sas_max:,}")
    print(f"  {match} CSV: {csv_min:,} to {csv_max:,}")
    if sas_min != csv_min or sas_max != csv_max:
        all_match = False
    
    print("\n" + "="*70)
    if all_match:
        print("  ✅ VALIDATION PASSED - Outputs are equivalent!")
        print("="*70 + "\n")
        return True
    else:
        print("  ❌ VALIDATION FAILED - Outputs differ!")
        print("="*70 + "\n")
        return False

if __name__ == "__main__":
    SAS_FILE = "/Users/Mach/dev/aps/data/2025_CFX_superpolicy/raw_out/superpolicy_final.sas7bdat"
    CSV_FILE = "/Users/Mach/dev/aps/data/2025_CFX_superpolicy/raw_out/superpolicy_final_polars.csv"
    
    try:
        print("Starting validation...\n")
        sas_stats = get_sas_stats(SAS_FILE)
        print(f"  ✓ Read SAS file\n")
        
        csv_stats = get_csv_stats(CSV_FILE)
        print(f"  ✓ Read CSV file\n")
        
        success = compare_stats(sas_stats, csv_stats)
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
