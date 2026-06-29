"""
Utility functions for N-fold data splitting.

This module contains all functions for:
- Configuration loading from JSON
- Data loading and preprocessing
- Objective function calculation
- Fold assignment
- Validation and plotting
"""

import pandas as pd
import numpy as np
import random
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def load_config(config_path="config.json"):
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config.json file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Resolve active EE columns based on selection
    if config['column_selection']['use_raw_ee']:
        config['ee_columns'] = config['column_mappings']['ee_columns_raw']
    else:
        config['ee_columns'] = config['column_mappings']['ee_columns_imps']
    
    # Resolve active incurred columns based on selection
    if config['column_selection']['use_raw_incurred']:
        config['incurred_columns'] = config['column_mappings']['incurred_columns_raw']
    else:
        config['incurred_columns'] = config['column_mappings']['incurred_columns_imps']
    
    # Set convenience variables
    config['join_key'] = config['column_mappings']['join_key']
    config['superpolicy_col'] = config['column_mappings']['superpolicy_col']
    config['all_coverages'] = config['coverages']['all_coverages']
    config['analysis_coverages'] = config['coverages']['analysis_coverages']
    config['objective_coverages'] = config['coverages']['objective_coverages']
    config['n_simulations'] = config['simulation_parameters']['n_simulations']
    config['n_folds'] = config['simulation_parameters']['n_folds']
    
    # Generate PP column names
    config['pp_columns'] = [f"pp_{cov}" for cov in config['analysis_coverages']]
    
    return config


# Global config - loaded when module is imported
CONFIG = load_config()


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_merge_data(config=None):
    """
    Load main data and superpolicy data, merge on join key.
    
    Args:
        config: Configuration dict (uses global CONFIG if None)
        
    Returns:
        pd.DataFrame: Merged dataframe
    """
    if config is None:
        config = CONFIG
    
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    # Load main data
    main_path = config['file_paths']['main_data_path']
    print(f"\nLoading main data from:\n  {main_path}")
    
    # Get columns to load - include all 6 coverages for analysis
    ee_cols = list(config['ee_columns'].values())
    inc_cols = list(config['incurred_columns'].values())
    columns_to_load = [config['join_key']] + ee_cols + inc_cols
    
    df_main = pd.read_parquet(main_path, columns=columns_to_load)
    print(f"  Main data shape: {df_main.shape}")
    
    # Load superpolicy data
    sp_path = config['file_paths']['superpolicy_path']
    print(f"\nLoading superpolicy data from:\n  {sp_path}")
    df_sp = pd.read_parquet(sp_path)
    print(f"  Superpolicy data shape: {df_sp.shape}")
    
    # Merge
    join_key = config['join_key']
    print(f"\nMerging on '{join_key}'...")
    df = pd.merge(df_main, df_sp, on=join_key, how='inner')
    print(f"  Merged data shape: {df.shape}")
    
    # Check for duplicates
    n_duplicates = df.duplicated(subset=join_key).sum()
    if n_duplicates > 0:
        print(f"  WARNING: {n_duplicates} duplicate {join_key} values found!")
    
    return df


def show_zero_exposure_stats(df, config=None, coverages=None):
    """
    Show statistics about zero exposure records without removing any.
    
    Args:
        df: Input dataframe
        config: Configuration dict (uses global CONFIG if None)
        coverages: List of coverages to check (uses all_coverages if None)
    """
    if config is None:
        config = CONFIG
    if coverages is None:
        coverages = config['all_coverages']
    
    print("\n" + "=" * 60)
    print("ZERO EXPOSURE STATISTICS (NO RECORDS REMOVED)")
    print("=" * 60)
    
    total_count = len(df)
    
    print(f"\nTotal records: {total_count:,}")
    print(f"\nZero exposure counts by coverage:")
    for cov in coverages:
        col = config['ee_columns'][cov]
        zero_count = (df[col] == 0).sum()
        nonzero_count = total_count - zero_count
        zero_pct = 100 * zero_count / total_count
        print(f"  {cov.upper():5s} ({col}): {zero_count:,} zeros ({zero_pct:.1f}%), {nonzero_count:,} non-zero")
    
    print(f"\n>>> All {total_count:,} records retained for fold assignment")


def remove_zero_exposure(df, config=None, coverages=None, remove=False):
    """
    Optionally remove records where any exposure column is zero for specified coverages.
    By default, just shows statistics without removing records.
    
    Args:
        df: Input dataframe
        config: Configuration dict (uses global CONFIG if None)
        coverages: List of coverages to check (uses analysis_coverages if None)
        remove: If True, actually remove records. If False (default), just show stats.
        
    Returns:
        pd.DataFrame: Original or cleaned dataframe depending on 'remove' parameter
    """
    if config is None:
        config = CONFIG
    if coverages is None:
        coverages = config['analysis_coverages']
    
    initial_count = len(df)
    
    if not remove:
        # Just show stats, don't remove
        show_zero_exposure_stats(df, config, coverages)
        return df
    
    print("\n" + "=" * 60)
    print("REMOVING ZERO EXPOSURE RECORDS")
    print("=" * 60)
    
    # Create mask for records with any zero exposure (only for specified coverages)
    ee_cols = [config['ee_columns'][cov] for cov in coverages]
    zero_mask = (df[ee_cols] == 0).any(axis=1)
    
    # Count by coverage
    print(f"\nZero exposure counts by coverage (checking: {coverages}):")
    for cov in coverages:
        col = config['ee_columns'][cov]
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


def calculate_pure_premium(df, config=None, coverages=None):
    """
    Calculate Pure Premium for each coverage.
    
    Args:
        df: Input dataframe with EE and incurred columns
        config: Configuration dict (uses global CONFIG if None)
        coverages: List of coverages to calculate PP for (uses analysis_coverages if None)
        
    Returns:
        pd.DataFrame: Dataframe with PP columns added
    """
    if config is None:
        config = CONFIG
    if coverages is None:
        coverages = config['analysis_coverages']
    
    print("\n" + "=" * 60)
    print("CALCULATING PURE PREMIUM")
    print("=" * 60)
    
    df = df.copy()
    
    for cov in coverages:
        ee_col = config['ee_columns'][cov]
        inc_col = config['incurred_columns'][cov]
        pp_col = f"pp_{cov}"
        
        df[pp_col] = df[inc_col] / df[ee_col]
        
        # Handle any infinities
        df[pp_col] = df[pp_col].replace([np.inf, -np.inf], np.nan)
    
    print("Pure Premium columns created:")
    for cov in coverages:
        pp_col = f"pp_{cov}"
        print(f"  {pp_col}: mean={df[pp_col].mean():.2f}, median={df[pp_col].median():.2f}")
    
    return df


# =============================================================================
# OBJECTIVE FUNCTION AND STATISTICS
# =============================================================================

def calculate_overall_stats(df, config=None, coverages=None):
    """
    Calculate overall Pure Premium and normalized EE weights.
    
    Args:
        df: Input dataframe
        config: Configuration dict (uses global CONFIG if None)
        coverages: List of coverages (uses objective_coverages if None)
        
    Returns:
        dict: Statistics by coverage
    """
    if config is None:
        config = CONFIG
    if coverages is None:
        coverages = config['objective_coverages']
    
    stats = {}
    total_ee = 0
    
    for cov in coverages:
        ee_col = config['ee_columns'][cov]
        inc_col = config['incurred_columns'][cov]
        
        total_ee_cov = df[ee_col].sum()
        total_inc_cov = df[inc_col].sum()
        
        stats[cov] = {
            'mean_pp': total_inc_cov / total_ee_cov if total_ee_cov > 0 else 0,
            'total_ee': total_ee_cov
        }
        total_ee += total_ee_cov
    
    # Calculate normalized weights
    for cov in coverages:
        stats[cov]['norm_ee'] = stats[cov]['total_ee'] / total_ee if total_ee > 0 else 0
    
    return stats


def print_overall_stats(overall_stats, coverages=None, config=None):
    """Print overall statistics in a formatted way."""
    if config is None:
        config = CONFIG
    if coverages is None:
        coverages = config['objective_coverages']
    
    print("\n--- Overall Statistics ---")
    for cov in coverages:
        if cov in overall_stats:
            print(f"  {cov.upper()}: PP={overall_stats[cov]['mean_pp']:.2f}, "
                  f"Weight={overall_stats[cov]['norm_ee']:.4f}")


def calculate_fold_pp(df, fold_num, config=None, coverages=None):
    """
    Calculate Pure Premium for a specific fold.
    
    Args:
        df: Dataframe with 'fold' column
        fold_num: Fold number (1-N)
        config: Configuration dict (uses global CONFIG if None)
        coverages: List of coverages (uses objective_coverages if None)
        
    Returns:
        dict: Pure Premium values by coverage
    """
    if config is None:
        config = CONFIG
    if coverages is None:
        coverages = config['objective_coverages']
    
    fold_data = df[df['fold'] == fold_num]
    
    pp_values = {}
    for cov in coverages:
        ee_col = config['ee_columns'][cov]
        inc_col = config['incurred_columns'][cov]
        
        total_ee = fold_data[ee_col].sum()
        total_inc = fold_data[inc_col].sum()
        
        pp_values[cov] = total_inc / total_ee if total_ee > 0 else 0
    
    return pp_values


def objective_function(fold_pp, overall_stats, coverages=None, config=None):
    """
    Calculate objective function value.
    Lower is better - measures how close fold PP is to overall PP.
    
    Uses objective_coverages from config (excludes med by default).
    
    Args:
        fold_pp: Dict of fold Pure Premium by coverage
        overall_stats: Overall statistics dict
        coverages: List of coverages to use (uses objective_coverages if None)
        config: Configuration dict (uses global CONFIG if None)
        
    Returns:
        float: Objective function value
    """
    if config is None:
        config = CONFIG
    if coverages is None:
        coverages = config['objective_coverages']
    
    objective = 0
    for cov in coverages:
        if cov in fold_pp and cov in overall_stats:
            diff = abs(fold_pp[cov] - overall_stats[cov]['mean_pp'])
            weight = overall_stats[cov]['norm_ee']
            objective += diff * weight
    
    return objective


def calculate_total_objective(df, overall_stats, config=None, coverages=None):
    """
    Calculate average objective across all folds.
    
    Args:
        df: Dataframe with 'fold' column
        overall_stats: Overall statistics dict
        config: Configuration dict (uses global CONFIG if None)
        coverages: List of coverages (uses objective_coverages if None)
        
    Returns:
        tuple: (average objective, list of fold objectives)
    """
    if config is None:
        config = CONFIG
    if coverages is None:
        coverages = config['objective_coverages']
    
    n_folds = config['n_folds']
    fold_objectives = []
    
    for fold_num in range(1, n_folds + 1):
        fold_pp = calculate_fold_pp(df, fold_num, config, coverages)
        obj = objective_function(fold_pp, overall_stats, coverages, config)
        fold_objectives.append(obj)
    
    return np.mean(fold_objectives), fold_objectives


# =============================================================================
# FOLD ASSIGNMENT
# =============================================================================

def assign_folds(df, seed, config=None):
    """
    Assign each record to a fold (1-N) based on superpolicy_id.
    
    Args:
        df: Input dataframe
        seed: Random seed for reproducibility
        config: Configuration dict (uses global CONFIG if None)
        
    Returns:
        pd.DataFrame: Dataframe with 'fold' column
    """
    if config is None:
        config = CONFIG
    
    df = df.copy()
    random.seed(seed)
    
    n_folds = config['n_folds']
    superpolicy_col = config['superpolicy_col']
    
    # Get unique superpolicies and shuffle
    unique_ids = df[superpolicy_col].unique().tolist()
    unique_ids_sorted = sorted(unique_ids)
    shuffled_ids = random.sample(unique_ids_sorted, len(unique_ids_sorted))
    
    # Divide into N_FOLDS
    fold_size = len(shuffled_ids) // n_folds
    
    id_to_fold = {}
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(shuffled_ids)
        for sp_id in shuffled_ids[start:end]:
            id_to_fold[sp_id] = i + 1  # Folds 1-N
    
    df['fold'] = df[superpolicy_col].map(id_to_fold)
    return df


# =============================================================================
# SIMULATION LOOP
# =============================================================================

def run_simulations(df, overall_stats, n_simulations=None, config=None, coverages=None):
    """
    Run simulations to find the best seed.
    
    Args:
        df: Input dataframe
        overall_stats: Overall statistics dict
        n_simulations: Number of seeds to try (default from config)
        config: Configuration dict (uses global CONFIG if None)
        coverages: List of coverages for objective (uses objective_coverages if None)
        
    Returns:
        pd.DataFrame: Results with seed, objective values, time
    """
    if config is None:
        config = CONFIG
    if n_simulations is None:
        n_simulations = config['n_simulations']
    if coverages is None:
        coverages = config['objective_coverages']
    
    n_folds = config['n_folds']
    
    print("\n" + "=" * 60)
    print(f"RUNNING {n_simulations} SIMULATIONS ({n_folds} folds)")
    print(f"Objective coverages: {coverages}")
    print("=" * 60)
    
    results = []
    start_time = time.time()
    
    for seed in range(1, n_simulations + 1):
        iter_start = time.time()
        
        # Assign folds with current seed
        df_temp = assign_folds(df, seed, config)
        
        # Calculate objective
        avg_obj, fold_objs = calculate_total_objective(df_temp, overall_stats, config, coverages)
        
        iter_time = time.time() - iter_start
        
        result = {
            'seed': seed,
            'avg_objective': avg_obj,
            'time_seconds': iter_time
        }
        
        # Add fold-specific objectives
        for i, obj in enumerate(fold_objs):
            result[f'fold_{i+1}_obj'] = obj
        
        results.append(result)
        
        # Progress update every 10 iterations
        if seed % 10 == 0:
            elapsed = time.time() - start_time
            best_result = min(results, key=lambda r: r['avg_objective'])
            best_seed_so_far = best_result['seed']
            best_obj_so_far = best_result['avg_objective']
            print(f"  Completed {seed}/{n_simulations} seeds... "
                  f"(elapsed: {elapsed:.1f}s, best so far: seed {best_seed_so_far} = {best_obj_so_far:.4f})")
    
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


def plot_all_coverage_histograms(df, config=None, coverages=None):
    """
    Plot histograms of all coverage EE columns.
    
    Args:
        df: Input dataframe
        config: Configuration dict (uses global CONFIG if None)
        coverages: List of coverages (uses all_coverages if None)
    """
    if config is None:
        config = CONFIG
    if coverages is None:
        coverages = config['all_coverages']
    
    n_cov = len(coverages)
    n_cols = 3
    n_rows = (n_cov + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    
    for i, cov in enumerate(coverages):
        ax = axes[i]
        col = config['ee_columns'][cov]
        
        # Get data
        data = df[col]
        
        # Stats
        zero_count = (data == 0).sum()
        zero_pct = 100 * zero_count / len(data)
        nonzero_data = data[data > 0]
        
        # Plot non-zero values only
        if len(nonzero_data) > 0:
            ax.hist(nonzero_data, bins=50, edgecolor='black', alpha=0.7)
            ax.set_title(f"{cov.upper()} (Zero: {zero_pct:.1f}%)\nmean={nonzero_data.mean():.2f}")
        else:
            ax.text(0.5, 0.5, f"100% Zero", ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f"{cov.upper()} - All Zero!")
        
        ax.set_xlabel(col)
    
    # Hide unused subplots
    for i in range(len(coverages), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Earned Exposure Distributions (Non-Zero Values Only)', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print("\nSummary of zero exposure:")
    for cov in coverages:
        col = config['ee_columns'][cov]
        zero_count = (df[col] == 0).sum()
        zero_pct = 100 * zero_count / len(df)
        print(f"  {cov.upper():5s}: {zero_count:,} zeros ({zero_pct:.1f}%)")


def plot_incurred_histograms(df, config=None, coverages=None):
    """
    Plot histograms of all coverage incurred columns.
    
    Args:
        df: Input dataframe
        config: Configuration dict (uses global CONFIG if None)
        coverages: List of coverages (uses all_coverages if None)
    """
    if config is None:
        config = CONFIG
    if coverages is None:
        coverages = config['all_coverages']
    
    n_cov = len(coverages)
    n_cols = 3
    n_rows = (n_cov + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    
    for i, cov in enumerate(coverages):
        ax = axes[i]
        col = config['incurred_columns'][cov]
        
        # Get data
        data = df[col]
        
        # Stats
        zero_count = (data == 0).sum()
        zero_pct = 100 * zero_count / len(data)
        nonzero_data = data[data > 0]
        
        # Plot non-zero values only
        if len(nonzero_data) > 0:
            ax.hist(nonzero_data, bins=50, edgecolor='black', alpha=0.7, color='orange')
            ax.set_title(f"{cov.upper()} (Zero: {zero_pct:.1f}%)\nmean={nonzero_data.mean():.2f}")
        else:
            ax.text(0.5, 0.5, f"100% Zero", ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f"{cov.upper()} - All Zero!")
        
        ax.set_xlabel(col)
    
    # Hide unused subplots
    for i in range(len(coverages), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Incurred Loss Distributions (Non-Zero Values Only)', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print("\nSummary of zero incurred:")
    for cov in coverages:
        col = config['incurred_columns'][cov]
        zero_count = (df[col] == 0).sum()
        zero_pct = 100 * zero_count / len(df)
        print(f"  {cov.upper():5s}: {zero_count:,} zeros ({zero_pct:.1f}%)")


def analyze_bi_pd_relationship(df, config=None):
    """
    Analyze BI vs PD relationship to determine if they should be combined.
    
    Args:
        df: Input dataframe
        config: Configuration dict (uses global CONFIG if None)
    """
    if config is None:
        config = CONFIG
    
    print("\n" + "=" * 60)
    print("BI vs PD RELATIONSHIP ANALYSIS")
    print("=" * 60)
    
    ee_bi = config['ee_columns']['bi']
    ee_pd = config['ee_columns']['pd']
    inc_bi = config['incurred_columns']['bi']
    inc_pd = config['incurred_columns']['pd']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. EE BI vs EE PD scatter
    ax = axes[0, 0]
    ax.scatter(df[ee_bi], df[ee_pd], alpha=0.1, s=1)
    ax.set_xlabel(f'BI Exposure ({ee_bi})')
    ax.set_ylabel(f'PD Exposure ({ee_pd})')
    ax.set_title('BI vs PD Exposure')
    
    # Add diagonal line
    max_val = max(df[ee_bi].max(), df[ee_pd].max())
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')
    ax.legend()
    
    # 2. EE Difference histogram
    ax = axes[0, 1]
    ee_diff = df[ee_bi] - df[ee_pd]
    ax.hist(ee_diff, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', label='Zero diff')
    ax.axvline(ee_diff.mean(), color='green', linestyle='--', label=f'Mean: {ee_diff.mean():.4f}')
    ax.set_xlabel('BI Exposure - PD Exposure')
    ax.set_ylabel('Frequency')
    ax.set_title('Exposure Difference (BI - PD)')
    ax.legend()
    
    # 3. Combined incurred loss histogram
    ax = axes[1, 0]
    combined_inc = df[inc_bi] + df[inc_pd]
    nonzero_combined = combined_inc[combined_inc > 0]
    if len(nonzero_combined) > 0:
        ax.hist(nonzero_combined, bins=50, edgecolor='black', alpha=0.7, color='purple')
        ax.set_title(f'BI+PD Combined Incurred\nmean={nonzero_combined.mean():.2f}')
    ax.set_xlabel('BI + PD Incurred Loss')
    ax.set_ylabel('Frequency')
    
    # 4. Comparison: BI PP vs PD PP
    ax = axes[1, 1]
    # Calculate PP for both
    bi_pp = df[inc_bi] / df[ee_bi]
    pd_pp = df[inc_pd] / df[ee_pd]
    bi_pp = bi_pp.replace([np.inf, -np.inf], np.nan)
    pd_pp = pd_pp.replace([np.inf, -np.inf], np.nan)
    
    # Plot histograms overlaid
    ax.hist(bi_pp.dropna(), bins=50, alpha=0.5, label=f'BI PP (mean={bi_pp.mean():.2f})', edgecolor='black')
    ax.hist(pd_pp.dropna(), bins=50, alpha=0.5, label=f'PD PP (mean={pd_pp.mean():.2f})', edgecolor='black')
    ax.set_xlabel('Pure Premium')
    ax.set_ylabel('Frequency')
    ax.set_title('BI vs PD Pure Premium Comparison')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n--- Summary Statistics ---")
    print(f"\nExposure (EE):")
    print(f"  BI mean: {df[ee_bi].mean():.4f}, sum: {df[ee_bi].sum():,.0f}")
    print(f"  PD mean: {df[ee_pd].mean():.4f}, sum: {df[ee_pd].sum():,.0f}")
    print(f"  Correlation: {df[ee_bi].corr(df[ee_pd]):.4f}")
    print(f"  Mean difference (BI-PD): {ee_diff.mean():.4f}")
    print(f"  Records where BI == PD: {(ee_diff == 0).sum():,} ({100*(ee_diff == 0).sum()/len(df):.1f}%)")
    
    print(f"\nIncurred Loss:")
    print(f"  BI mean: {df[inc_bi].mean():.2f}, sum: {df[inc_bi].sum():,.0f}")
    print(f"  PD mean: {df[inc_pd].mean():.2f}, sum: {df[inc_pd].sum():,.0f}")
    print(f"  BI+PD mean: {combined_inc.mean():.2f}, sum: {combined_inc.sum():,.0f}")
    
    print(f"\nPure Premium:")
    print(f"  BI PP mean: {bi_pp.mean():.2f}")
    print(f"  PD PP mean: {pd_pp.mean():.2f}")
    print(f"  Combined PP (if BI+PD used): {combined_inc.sum() / df[ee_bi].sum():.2f}")
    
    print("\n--- Recommendation ---")
    corr = df[ee_bi].corr(df[ee_pd])
    same_pct = 100 * (ee_diff == 0).sum() / len(df)
    
    if same_pct > 90:
        print("  BI and PD exposures are nearly identical (>90% same).")
        print("  Consider treating them as the SAME exposure for fold stratification.")
    elif corr > 0.95:
        print(f"  BI and PD exposures are highly correlated ({corr:.4f}).")
        print("  They could be combined or used interchangeably.")
    else:
        print(f"  BI and PD exposures have moderate correlation ({corr:.4f}).")
        print("  Recommend treating them SEPARATELY in the objective function.")


# =============================================================================
# VALIDATION
# =============================================================================

def validate_folds(df, overall_stats, config=None, coverages=None):
    """
    Validate the fold assignment quality.
    
    Args:
        df: Dataframe with 'fold' column
        overall_stats: Overall statistics dict
        config: Configuration dict (uses global CONFIG if None)
        coverages: List of coverages (uses objective_coverages if None)
    """
    if config is None:
        config = CONFIG
    if coverages is None:
        coverages = config['objective_coverages']
    
    n_folds = config['n_folds']
    superpolicy_col = config['superpolicy_col']
    
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
    sp_counts = df.groupby('fold')[superpolicy_col].nunique()
    total_sp = df[superpolicy_col].nunique()
    for fold, count in sp_counts.items():
        pct = 100 * count / total_sp
        print(f"  Fold {fold}: {count:,} superpolicies ({pct:.1f}%)")
    
    print("\n--- Pure Premium by Fold vs Overall ---")
    header = f"{'Coverage':<10} {'Overall':>12} " + " ".join([f"{'Fold '+str(i):>10}" for i in range(1, n_folds+1)])
    print(header)
    print("-" * len(header))
    
    for cov in coverages:
        overall_pp = overall_stats[cov]['mean_pp']
        fold_pps = []
        for fold_num in range(1, n_folds + 1):
            fold_pp = calculate_fold_pp(df, fold_num, config, coverages)
            fold_pps.append(fold_pp[cov])
        
        row = f"{cov.upper():<10} {overall_pp:>12.2f} " + " ".join([f"{pp:>10.2f}" for pp in fold_pps])
        print(row)
    
    print("\n--- Objective Function by Fold ---")
    for fold_num in range(1, n_folds + 1):
        fold_pp = calculate_fold_pp(df, fold_num, config, coverages)
        obj = objective_function(fold_pp, overall_stats, coverages, config)
        print(f"  Fold {fold_num}: {obj:.4f}")
    
    avg_obj, _ = calculate_total_objective(df, overall_stats, config, coverages)
    print(f"\n  Average Objective: {avg_obj:.4f}")


# =============================================================================
# OUTPUT
# =============================================================================

def save_output(df, output_path=None, simulation_results=None, sim_results_path=None, config=None):
    """
    Save fold assignments and simulation results.
    
    Args:
        df: Dataframe with fold assignments and PP columns
        output_path: Path for fold assignments parquet (default from config)
        simulation_results: Optional DataFrame of simulation results
        sim_results_path: Path for simulation results CSV (default from config)
        config: Configuration dict (uses global CONFIG if None)
    """
    if config is None:
        config = CONFIG
    if output_path is None:
        output_path = config['file_paths']['output_path']
    if sim_results_path is None:
        sim_results_path = config['file_paths']['simulation_results_path']
    
    # Create output directory if needed
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save simulation results
    if simulation_results is not None:
        simulation_results.to_csv(sim_results_path, index=False)
        print(f"\nSimulation results saved to:\n  {sim_results_path}")
    
    # Select output columns and save
    join_key = config['join_key']
    superpolicy_col = config['superpolicy_col']
    pp_columns = config['pp_columns']
    
    output_cols = [join_key, superpolicy_col, 'fold'] + pp_columns
    # Only include columns that exist in df
    output_cols = [c for c in output_cols if c in df.columns]
    df_output = df[output_cols].copy()
    
    df_output.to_parquet(output_path, index=False)
    print(f"\nFold assignments saved to:\n  {output_path}")
    print(f"  Shape: {df_output.shape}")
    print(f"  Columns: {list(df_output.columns)}")
    
    return df_output
