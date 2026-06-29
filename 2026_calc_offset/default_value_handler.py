"""
Default Value Handler Module
============================
This module provides functions to:
1. Check for missing/null/zero/empty values in key features
2. Fill missing values using lookup tables and defaults

Missing Value Logic:
- Odometer: Lookup by vehicle age from Dep_Model_Default_Odometer.csv
            For ages > 25, use the value for age 25
- State: Lookup by BSST from Dep_Model_Default_State_by_BSST.csv
- Population Density Percentile: Use 50 (median) as default
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


def check_missing_values(df: pd.DataFrame, model_year: int = 2026) -> pd.DataFrame:
    """
    Check for zeros, nulls, empty strings, and NaN for key features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with raw data
    model_year : int
        Model year for calculating CALC_VEH_AGE
        
    Returns:
    --------
    pd.DataFrame
        Summary dataframe with missing value counts
    """
    # Define features to check with their column mappings
    features = {
        'ODOMETER': 'cef_est_curr_mi_grp_imps',
        'geo_pop_density_ntile': 'zip_pop_dens',
        'CALC_VEH_AGE (dml_year_imps)': 'dml_year_imps',
        'STATE': 'st_raw',
        'BSST': 'BODY_STYLE_SEGMENT_BODY_TYPE'
    }
    
    total_rows = len(df)
    print(f"\n{'='*100}")
    print(f"DATA QUALITY CHECK - Missing/Zero/Null Values")
    print(f"{'='*100}")
    print(f"Total Records: {total_rows:,}\n")
    
    results = []
    
    print(f"{'Feature':<40} {'Zeros':>12} {'Null/NaN':>12} {'Empty Str':>12} {'Total Issue':>15} {'%':>10}")
    print("-" * 103)
    
    for feature_name, col_name in features.items():
        if col_name not in df.columns:
            print(f"{feature_name:<40} {'N/A - column not found':<60}")
            results.append({
                'Feature': feature_name,
                'Column': col_name,
                'Zeros': None,
                'Null_NaN': None,
                'Empty_String': None,
                'Total_Issues': None,
                'Percent': None
            })
            continue
        
        col = df[col_name]
        
        # Count zeros (only for numeric columns)
        if pd.api.types.is_numeric_dtype(col):
            zeros = (col == 0).sum()
        else:
            zeros = 0
        
        # Count null/NaN
        nulls = col.isna().sum()
        
        # Count empty strings (only for object/string columns)
        if col.dtype == 'object' or pd.api.types.is_string_dtype(col):
            empty = (col == '').sum()
        else:
            empty = 0
        
        # Total issues (zeros are only counted as issues for ODOMETER)
        if feature_name == 'ODOMETER':
            total_issues = zeros + nulls + empty
        else:
            total_issues = nulls + empty
        
        pct = (total_issues / total_rows) * 100 if total_rows > 0 else 0
        
        print(f"{feature_name:<40} {zeros:>12,} {nulls:>12,} {empty:>12,} {total_issues:>15,} {pct:>9.4f}%")
        
        results.append({
            'Feature': feature_name,
            'Column': col_name,
            'Zeros': zeros,
            'Null_NaN': nulls,
            'Empty_String': empty,
            'Total_Issues': total_issues,
            'Percent': pct
        })
    
    print("-" * 103)
    print(f"\nNote: For ODOMETER, zeros are counted as missing values (to be filled with defaults)")
    print(f"      For other features, only Null/NaN and Empty Strings are considered missing")
    
    return pd.DataFrame(results)


def load_odometer_defaults(filepath: str) -> pd.DataFrame:
    """
    Load odometer default values from CSV.
    
    Parameters:
    -----------
    filepath : str
        Path to Dep_Model_Default_Odometer.csv
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with CALC_VEH_AGE and ODOMETER_mean columns
    """
    df = pd.read_csv(filepath)
    print(f"✓ Loaded odometer defaults: {len(df)} vehicle age values (0-{df['CALC_VEH_AGE'].max()})")
    return df


def load_state_defaults(filepath: str) -> pd.DataFrame:
    """
    Load state default values from CSV.
    
    Parameters:
    -----------
    filepath : str
        Path to Dep_Model_Default_State_by_BSST.csv
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with BSST and STATE_mode columns
    """
    df = pd.read_csv(filepath)
    print(f"✓ Loaded state defaults: {len(df)} BSST types")
    return df


def fill_missing_odometer(
    df: pd.DataFrame,
    odometer_defaults: pd.DataFrame,
    model_year: int,
    odometer_col: str = 'cef_est_curr_mi_grp_imps',
    year_col: str = 'dml_year_imps',
    max_age: int = 25
) -> Tuple[pd.DataFrame, int]:
    """
    Fill missing odometer values (zeros) using vehicle age lookup.
    
    For vehicles with ages greater than max_age, use the default value for max_age.
    Also creates ODOMETER_IMP_FLAG column (1 = imputed, 0 = original).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    odometer_defaults : pd.DataFrame
        Lookup table with CALC_VEH_AGE and ODOMETER_mean
    model_year : int
        Model year for calculating vehicle age
    odometer_col : str
        Column name for odometer values
    year_col : str
        Column name for vehicle model year
    max_age : int
        Maximum age to use for lookup (default: 25)
        
    Returns:
    --------
    Tuple[pd.DataFrame, int]
        Updated dataframe (with ODOMETER_IMP_FLAG) and count of filled values
    """
    df = df.copy()
    
    # Calculate vehicle age
    df['CALC_VEH_AGE'] = model_year - df[year_col]
    
    # Cap age at max_age for lookup purposes
    df['_age_for_lookup'] = df['CALC_VEH_AGE'].clip(lower=0, upper=max_age)
    
    # Identify missing odometer (zeros or null)
    missing_mask = (df[odometer_col] == 0) | df[odometer_col].isna()
    count_before = missing_mask.sum()
    
    # Create ODOMETER_IMP_FLAG BEFORE filling (1 = will be imputed, 0 = original value)
    df['ODOMETER_IMP_FLAG'] = missing_mask.astype(int)
    
    if count_before == 0:
        print(f"✓ No missing odometer values to fill")
        print(f"✓ Created ODOMETER_IMP_FLAG column (all zeros - no imputation needed)")
        df.drop(columns=['_age_for_lookup'], inplace=True)
        return df, 0
    
    # Create lookup dictionary for faster mapping
    odo_lookup = odometer_defaults.set_index('CALC_VEH_AGE')['ODOMETER_mean'].to_dict()
    
    # Fill missing values using vectorized operation
    df.loc[missing_mask, odometer_col] = df.loc[missing_mask, '_age_for_lookup'].map(odo_lookup)
    
    # Clean up temporary column
    df.drop(columns=['_age_for_lookup'], inplace=True)
    
    count_filled = count_before - ((df[odometer_col] == 0) | df[odometer_col].isna()).sum()
    
    print(f"✓ Filled {count_filled:,} missing odometer values using vehicle age lookup")
    print(f"  (Ages > {max_age} used the default for age {max_age}: {odo_lookup.get(max_age, 'N/A'):,.0f} miles)")
    print(f"✓ Created ODOMETER_IMP_FLAG column ({df['ODOMETER_IMP_FLAG'].sum():,} records flagged as imputed)")
    
    return df, count_filled


def fill_missing_state(
    df: pd.DataFrame,
    state_defaults: pd.DataFrame,
    state_col: str = 'st_raw',
    bsst_col: str = 'BODY_STYLE_SEGMENT_BODY_TYPE'
) -> Tuple[pd.DataFrame, int]:
    """
    Fill missing state values using BSST lookup.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    state_defaults : pd.DataFrame
        Lookup table with BSST and STATE_mode
    state_col : str
        Column name for state values
    bsst_col : str
        Column name for BSST values
        
    Returns:
    --------
    Tuple[pd.DataFrame, int]
        Updated dataframe and count of filled values
    """
    df = df.copy()
    
    # Identify missing state (null or empty string)
    missing_mask = df[state_col].isna() | (df[state_col] == '')
    count_before = missing_mask.sum()
    
    if count_before == 0:
        print(f"✓ No missing state values to fill")
        return df, 0
    
    # Create lookup dictionary
    state_lookup = state_defaults.set_index('BSST')['STATE_mode'].to_dict()
    
    # Fill missing values using BSST lookup
    df.loc[missing_mask, state_col] = df.loc[missing_mask, bsst_col].map(state_lookup)
    
    # Count how many were actually filled (some may still be missing if BSST not in lookup)
    still_missing = df[state_col].isna() | (df[state_col] == '')
    count_filled = count_before - still_missing.sum()
    
    print(f"✓ Filled {count_filled:,} missing state values using BSST lookup")
    if still_missing.sum() > 0:
        print(f"  ⚠ {still_missing.sum():,} records still have missing state (BSST not in lookup table)")
    
    return df, count_filled


def fill_missing_pop_density(
    df: pd.DataFrame,
    default_value: int = 50,
    pop_density_col: str = 'zip_pop_dens'
) -> Tuple[pd.DataFrame, int]:
    """
    Fill missing population density percentile with default value (median).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    default_value : int
        Default value to use (default: 50 = median)
    pop_density_col : str
        Column name for population density percentile
        
    Returns:
    --------
    Tuple[pd.DataFrame, int]
        Updated dataframe and count of filled values
    """
    df = df.copy()
    
    # Identify missing values (null/NaN)
    missing_mask = df[pop_density_col].isna()
    count_before = missing_mask.sum()
    
    if count_before == 0:
        print(f"✓ No missing population density percentile values to fill")
        return df, 0
    
    # Fill with default value
    df.loc[missing_mask, pop_density_col] = default_value
    
    print(f"✓ Filled {count_before:,} missing population density percentile values with default: {default_value}")
    
    return df, count_before


def apply_all_defaults(
    df: pd.DataFrame,
    config: Dict,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Apply all default value logic to the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with raw data
    config : Dict
        Configuration dictionary with paths and settings
    verbose : bool
        Whether to print detailed output
        
    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, int]]
        Updated dataframe and dictionary with counts of filled values
    """
    fill_counts = {}
    
    if verbose:
        print(f"\n{'='*100}")
        print(f"APPLYING DEFAULT VALUES")
        print(f"{'='*100}")
    
    # Load lookup tables
    odometer_defaults = load_odometer_defaults(config['default_odometer_path'])
    state_defaults = load_state_defaults(config['default_state_by_bsst_path'])
    
    print()
    
    # 1. Fill missing odometer
    df, count = fill_missing_odometer(
        df, 
        odometer_defaults, 
        model_year=config['model_year']
    )
    fill_counts['odometer'] = count
    
    # 2. Fill missing state
    df, count = fill_missing_state(df, state_defaults)
    fill_counts['state'] = count
    
    # 3. Fill missing population density percentile
    df, count = fill_missing_pop_density(
        df, 
        default_value=config['default_pop_density_percentile']
    )
    fill_counts['pop_density'] = count
    
    if verbose:
        print(f"\n{'='*100}")
        print(f"DEFAULT VALUES APPLIED SUCCESSFULLY")
        total_filled = sum(fill_counts.values())
        print(f"Total values filled: {total_filled:,}")
        print(f"{'='*100}")
    
    return df, fill_counts


# Main execution for testing
if __name__ == '__main__':
    import json
    
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    print("Default Value Handler - Test Run")
    print("="*50)
    
    # Load a sample of data for testing
    print("\nLoading sample data...")
    df = pd.read_parquet(config['car_parquet_path'], columns=[
        'cef_est_curr_mi_grp_imps',
        'zip_pop_dens',
        'dml_year_imps',
        'st_raw',
        'vin'
    ])
    
    # Load and merge BSST
    df_bsst = pd.read_parquet(config['vin_bsst_path'], columns=['VIN', 'BODY_STYLE_SEGMENT_BODY_TYPE'])
    df = df.merge(df_bsst, left_on='vin', right_on='VIN', how='left')
    
    # Check missing values before
    print("\n--- BEFORE APPLYING DEFAULTS ---")
    check_missing_values(df, config['model_year'])
    
    # Apply defaults
    df, fill_counts = apply_all_defaults(df, config)
    
    # Check missing values after
    print("\n--- AFTER APPLYING DEFAULTS ---")
    check_missing_values(df, config['model_year'])
