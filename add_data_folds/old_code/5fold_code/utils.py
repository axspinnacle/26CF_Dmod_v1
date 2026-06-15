"""
Utility functions for 5-fold data splitting.

This module contains all functions for:
- Data loading and preprocessing
- Objective function calculation
- Fold assignment
- Validation and plotting
"""

import pandas as pd
import numpy as np
import random
import time
from pathlib import Path
import matplotlib.pyplot as plt

from config import (
    MAIN_DATA_PATH, SUPERPOLICY_PATH, OUTPUT_PATH, SIMULATION_RESULTS_PATH,
    JOIN_KEY, SUPERPOLICY_COL, EE_COLUMNS, INCURRED_COLUMNS, COVERAGES,
    N_SIMULATIONS, N_FOLDS, PP_COLUMNS
)


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_merge_data():
    """
    Load main data and superpolicy data, merge on vin_date.
    
    Returns:
        pd.DataFrame: Merged dataframe
    """
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    # Load main data
    print(f"\nLoading main data from:\n  {MAIN_DATA_PATH}")
    ee_cols = list(EE_COLUMNS.values())
    inc_cols = list(INCURRED_COLUMNS.values())
    columns_to_load = [JOIN_KEY] + ee_cols + inc_cols
    
    df_main = pd.read_parquet(MAIN_DATA_PATH, columns=columns_to_load)
    print(f"  Main data shape: {df_main.shape}")
    
    # Load superpolicy data
    print(f"\nLoading superpolicy data from:\n  {SUPERPOLICY_PATH}")
    df_sp = pd.read_parquet(SUPERPOLICY_PATH)
    print(f"  Superpolicy data shape: {df_sp.shape}")
    
    # Merge
    print(f"\nMerging on '{JOIN_KEY}'...")
    df = pd.merge(df_main, df_sp, on=JOIN_KEY, how='inner')
    print(f"  Merged data shape: {df.shape}")
    
    # Check for duplicates
    n_duplicates = df.duplicated(subset=JOIN_KEY).sum()
    if n_duplicates > 0:
        print(f"  WARNING: {n_duplicates} duplicate {JOIN_KEY} values found!")
    
    return df


def remove_zero_exposure(df):
    """
    Remove records where any exposure column is zero.
    
    Args:
        df: Input dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print("\n" + "=" * 60)
    print("REMOVING ZERO EXPOSURE RECORDS")
    print("=" * 60)
    
    initial_count = len(df)
    
    # Create mask for records with any zero exposure
    ee_cols = list(EE_COLUMNS.values())
    zero_mask = (df[ee_cols] == 0).any(axis=1)
    
    # Count by coverage
    print("\nZero exposure counts by coverage:")
    for cov, col in EE_COLUMNS.items():
        zero_count = (df[col] == 0).sum()
        zero_pct = 100 * zero_count / initial_count
        print(f"  {cov.upper():5s} ({col}): {zero_count:,} records ({zero_pct:.2f}%)")
    
    # Remove records
    df_clean = df[~zero_mask].copy()
    
    removed_count = initial_count - len(df_clean)
    removed_pct = 100 * removed_count / initial_count
    
    print(f"\n>>> Removed {removed_count:,} records ({removed_pct:.2f}%) with zero exposure")
    print(f">>> Remaining: {len(df_clean):,} records")
    
    return df_clean


def calculate_pure_premium(df):
    """
    Calculate Pure Premium for each coverage.
    
    Args:
        df: Input dataframe with EE and incurred columns
        
    Returns:
        pd.DataFrame: Dataframe with PP columns added
    """
    print("\n" + "=" * 60)
    print("CALCULATING PURE PREMIUM")
    print("=" * 60)
    
    df = df.copy()
    
    for cov in COVERAGES:
        ee_col = EE_COLUMNS[cov]
        inc_col = INCURRED_COLUMNS[cov]
        pp_col = f"pp_{cov}"
        
        df[pp_col] = df[inc_col] / df[ee_col]
        
        # Handle any infinities
        df[pp_col] = df[pp_col].replace([np.inf, -np.inf], np.nan)
    
    print("Pure Premium columns created:")
    for cov in COVERAGES:
        pp_col = f"pp_{cov}"
        print(f"  {pp_col}: mean={df[pp_col].mean():.2f}, median={df[pp_col].median():.2f}")
    
    return df


# =============================================================================
# OBJECTIVE FUNCTION AND STATISTICS
# =============================================================================

def calculate_overall_stats(df):
    """
    Calculate overall Pure Premium and normalized EE weights.
    
    Args:
        df: Input dataframe
        
    Returns:
        dict: Statistics by coverage
    """
    stats = {}
    total_ee = 0
    
    for cov in COVERAGES:
        ee_col = EE_COLUMNS[cov]
        inc_col = INCURRED_COLUMNS[cov]
        
        total_ee_cov = df[ee_col].sum()
        total_inc_cov = df[inc_col].sum()
        
        stats[cov] = {
            'mean_pp': total_inc_cov / total_ee_cov if total_ee_cov > 0 else 0,
            'total_ee': total_ee_cov
        }
        total_ee += total_ee_cov
    
    # Calculate normalized weights
    for cov in COVERAGES:
        stats[cov]['norm_ee'] = stats[cov]['total_ee'] / total_ee if total_ee > 0 else 0
    
    return stats


def print_overall_stats(overall_stats):
    """Print overall statistics in a formatted way."""
    print("\n--- Overall Statistics ---")
    for cov in COVERAGES:
        print(f"  {cov.upper()}: PP={overall_stats[cov]['mean_pp']:.2f}, "
              f"Weight={overall_stats[cov]['norm_ee']:.4f}")


def calculate_fold_pp(df, fold_num):
    """
    Calculate Pure Premium for a specific fold.
    
    Args:
        df: Dataframe with 'fold' column
        fold_num: Fold number (1-5)
        
    Returns:
        dict: Pure Premium values by coverage
    """
    fold_data = df[df['fold'] == fold_num]
    
    pp_values = {}
    for cov in COVERAGES:
        ee_col = EE_COLUMNS[cov]
        inc_col = INCURRED_COLUMNS[cov]
        
        total_ee = fold_data[ee_col].sum()
        total_inc = fold_data[inc_col].sum()
        
        pp_values[cov] = total_inc / total_ee if total_ee > 0 else 0
    
    return pp_values


def objective_function(fold_pp, overall_stats):
    """
    Calculate objective function value.
    Lower is better - measures how close fold PP is to overall PP.
    
    Args:
        fold_pp: Dict of fold Pure Premium by coverage
        overall_stats: Overall statistics dict
        
    Returns:
        float: Objective function value
    """
    objective = 0
    for cov in COVERAGES:
        diff = abs(fold_pp[cov] - overall_stats[cov]['mean_pp'])
        weight = overall_stats[cov]['norm_ee']
        objective += diff * weight
    
    return objective


def calculate_total_objective(df, overall_stats):
    """
    Calculate average objective across all folds.
    
    Args:
        df: Dataframe with 'fold' column
        overall_stats: Overall statistics dict
        
    Returns:
        tuple: (average objective, list of fold objectives)
    """
    fold_objectives = []
    
    for fold_num in range(1, N_FOLDS + 1):
        fold_pp = calculate_fold_pp(df, fold_num)
        obj = objective_function(fold_pp, overall_stats)
        fold_objectives.append(obj)
    
    return np.mean(fold_objectives), fold_objectives


# =============================================================================
# FOLD ASSIGNMENT
# =============================================================================

def assign_folds(df, seed):
    """
    Assign each record to a fold (1-5) based on superpolicy_id.
    
    Args:
        df: Input dataframe
        seed: Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Dataframe with 'fold' column
    """
    df = df.copy()
    random.seed(seed)
    
    # Get unique superpolicies and shuffle
    unique_ids = df[SUPERPOLICY_COL].unique().tolist()
    unique_ids_sorted = sorted(unique_ids)
    shuffled_ids = random.sample(unique_ids_sorted, len(unique_ids_sorted))
    
    # Divide into N_FOLDS
    fold_size = len(shuffled_ids) // N_FOLDS
    
    id_to_fold = {}
    for i in range(N_FOLDS):
        start = i * fold_size
        end = start + fold_size if i < N_FOLDS - 1 else len(shuffled_ids)
        for sp_id in shuffled_ids[start:end]:
            id_to_fold[sp_id] = i + 1  # Folds 1-5
    
    df['fold'] = df[SUPERPOLICY_COL].map(id_to_fold)
    return df


# =============================================================================
# SIMULATION LOOP
# =============================================================================

def run_simulations(df, overall_stats, n_simulations=None):
    """
    Run N_SIMULATIONS to find the best seed.
    
    Args:
        df: Input dataframe
        overall_stats: Overall statistics dict
        n_simulations: Number of seeds to try (default from config)
        
    Returns:
        pd.DataFrame: Results with seed, objective values, time
    """
    if n_simulations is None:
        n_simulations = N_SIMULATIONS
        
    print("\n" + "=" * 60)
    print(f"RUNNING {n_simulations} SIMULATIONS")
    print("=" * 60)
    
    results = []
    start_time = time.time()
    
    for seed in range(1, n_simulations + 1):
        iter_start = time.time()
        
        # Assign folds with current seed
        df_temp = assign_folds(df, seed)
        
        # Calculate objective
        avg_obj, fold_objs = calculate_total_objective(df_temp, overall_stats)
        
        iter_time = time.time() - iter_start
        
        results.append({
            'seed': seed,
            'avg_objective': avg_obj,
            'fold_1_obj': fold_objs[0],
            'fold_2_obj': fold_objs[1],
            'fold_3_obj': fold_objs[2],
            'fold_4_obj': fold_objs[3],
            'fold_5_obj': fold_objs[4],
            'time_seconds': iter_time
        })
        
        # Progress update every 10 iterations
        if seed % 10 == 0:
            elapsed = time.time() - start_time
            best_so_far = min(r['avg_objective'] for r in results)
            print(f"  Completed {seed}/{n_simulations} seeds... "
                  f"(elapsed: {elapsed:.1f}s, best so far: {best_so_far:.4f})")
    
    total_time = time.time() - start_time
    print(f"\nTotal simulation time: {total_time:.1f} seconds")
    
    return pd.DataFrame(results)


def get_best_seed(results_df):
    """
    Find the best seed from simulation results.
    
    Args:
        results_df: DataFrame from run_simulations
        
    Returns:
        tuple: (best_seed, best_objective)
    """
    best_idx = results_df['avg_objective'].idxmin()
    best_seed = int(results_df.loc[best_idx, 'seed'])
    best_obj = results_df.loc[best_idx, 'avg_objective']
    
    print(f"\n>>> BEST SEED: {best_seed} (Objective: {best_obj:.4f})")
    
    return best_seed, best_obj


# =============================================================================
# PLOTTING
# =============================================================================

def plot_objective_distribution(results_df, save_path=None):
    """
    Plot histogram of objective function values.
    
    Args:
        results_df: DataFrame from run_simulations
        save_path: Optional path to save the plot
    """
    print("\n" + "=" * 60)
    print("OBJECTIVE FUNCTION DISTRIBUTION")
    print("=" * 60)
    
    objectives = results_df['avg_objective']
    
    print(f"\nStatistics:")
    print(f"  Min:    {objectives.min():.4f}")
    print(f"  Max:    {objectives.max():.4f}")
    print(f"  Mean:   {objectives.mean():.4f}")
    print(f"  Median: {objectives.median():.4f}")
    print(f"  Std:    {objectives.std():.4f}")
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(objectives, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(objectives.min(), color='red', linestyle='--', 
                label=f'Best: {objectives.min():.4f}')
    plt.axvline(objectives.mean(), color='green', linestyle='--', 
                label=f'Mean: {objectives.mean():.4f}')
    plt.xlabel('Average Objective Function Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Objective Function Values Across Seeds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nHistogram saved to: {save_path}")
    
    plt.show()


# =============================================================================
# VALIDATION
# =============================================================================

def validate_folds(df, overall_stats):
    """
    Validate the fold assignment quality.
    
    Args:
        df: Dataframe with 'fold' column
        overall_stats: Overall statistics dict
    """
    print("\n" + "=" * 60)
    print("FOLD VALIDATION")
    print("=" * 60)
    
    print("\n--- Record Counts ---")
    fold_counts = df['fold'].value_counts().sort_index()
    total = len(df)
    for fold, count in fold_counts.items():
        pct = 100 * count / total
        print(f"  Fold {fold}: {count:,} records ({pct:.1f}%)")
    
    print("\n--- Superpolicy Counts ---")
    sp_counts = df.groupby('fold')[SUPERPOLICY_COL].nunique()
    total_sp = df[SUPERPOLICY_COL].nunique()
    for fold, count in sp_counts.items():
        pct = 100 * count / total_sp
        print(f"  Fold {fold}: {count:,} superpolicies ({pct:.1f}%)")
    
    print("\n--- Pure Premium by Fold vs Overall ---")
    header = f"{'Coverage':<10} {'Overall':>12} " + " ".join([f"{'Fold '+str(i):>10}" for i in range(1, N_FOLDS+1)])
    print(header)
    print("-" * len(header))
    
    for cov in COVERAGES:
        overall_pp = overall_stats[cov]['mean_pp']
        fold_pps = []
        for fold_num in range(1, N_FOLDS + 1):
            fold_pp = calculate_fold_pp(df, fold_num)
            fold_pps.append(fold_pp[cov])
        
        row = f"{cov.upper():<10} {overall_pp:>12.2f} " + " ".join([f"{pp:>10.2f}" for pp in fold_pps])
        print(row)
    
    print("\n--- Objective Function by Fold ---")
    for fold_num in range(1, N_FOLDS + 1):
        fold_pp = calculate_fold_pp(df, fold_num)
        obj = objective_function(fold_pp, overall_stats)
        print(f"  Fold {fold_num}: {obj:.4f}")
    
    avg_obj, _ = calculate_total_objective(df, overall_stats)
    print(f"\n  Average Objective: {avg_obj:.4f}")


# =============================================================================
# OUTPUT
# =============================================================================

def save_output(df, output_path=None, simulation_results=None, sim_results_path=None):
    """
    Save fold assignments and simulation results.
    
    Args:
        df: Dataframe with fold assignments and PP columns
        output_path: Path for fold assignments parquet (default from config)
        simulation_results: Optional DataFrame of simulation results
        sim_results_path: Path for simulation results CSV (default from config)
    """
    if output_path is None:
        output_path = OUTPUT_PATH
    if sim_results_path is None:
        sim_results_path = SIMULATION_RESULTS_PATH
    
    # Create output directory if needed
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save simulation results
    if simulation_results is not None:
        simulation_results.to_csv(sim_results_path, index=False)
        print(f"\nSimulation results saved to:\n  {sim_results_path}")
    
    # Select output columns and save
    output_cols = [JOIN_KEY, SUPERPOLICY_COL, 'fold'] + PP_COLUMNS
    df_output = df[output_cols].copy()
    
    df_output.to_parquet(output_path, index=False)
    print(f"\nFold assignments saved to:\n  {output_path}")
    print(f"  Shape: {df_output.shape}")
    print(f"  Columns: {list(df_output.columns)}")
    
    return df_output
