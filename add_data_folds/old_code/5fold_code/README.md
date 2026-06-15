# 5-Fold Data Splitting Code

This folder contains the Python code to create and validate 5-fold cross-validation splits for insurance modeling data.

## Files

| File | Description |
|------|-------------|
| `create_5folds.ipynb` | **Main notebook** - Interactive workflow for creating folds |
| `utils.py` | Utility functions (data loading, objective function, validation, etc.) |
| `config.py` | Configuration file (file paths, column mappings, parameters) |
| `README.md` | This documentation file |

## Quick Start

1. Open `create_5folds.ipynb` in Jupyter Notebook or VS Code
2. Run cells sequentially
3. Output will be saved to the configured path

## Input Files

| File | Path |
|------|------|
| Main Data | `/Users/Mach/dev/aps/data/2026_Dmodel_data/master_dataset_car.parquet` |
| Superpolicy | `/Users/Mach/dev/aps/data/2026_Dmodel_data/superpolicy/superpolicy_car.parquet` |

## Output Files

| File | Path |
|------|------|
| Fold Assignments | `/Users/Mach/dev/aps/data/2026_Dmodel_data/fold_superpolicy/car_master_dataset_fold_sp.parquet` |
| Simulation Results | `/Users/Mach/dev/aps/data/2026_Dmodel_data/fold_superpolicy/simulation_results.csv` |

## Output Columns

The output parquet file contains:

| Column | Description |
|--------|-------------|
| `vin_date` | Unique record identifier |
| `superpolicy_id` | Superpolicy grouping ID |
| `fold` | Fold assignment (1-5) |
| `pp_bi` | Pure Premium - Bodily Injury |
| `pp_pd` | Pure Premium - Property Damage |
| `pp_pip` | Pure Premium - Personal Injury Protection |
| `pp_med` | Pure Premium - Medical |
| `pp_coll` | Pure Premium - Collision |
| `pp_comp` | Pure Premium - Comprehensive |

## Notebook Workflow

The `create_5folds.ipynb` notebook has the following sections:

1. **Setup and Imports** - Load libraries and functions
2. **Load and Merge Data** - Read parquet files and merge
3. **Remove Zero Exposure Records** - Clean data (shows count/percentage removed)
4. **Calculate Pure Premium** - Add PP columns for each coverage
5. **Run Simulations** - Test 100 seeds to find optimal split
6. **View Distribution** - Histogram of objective function values
7. **Apply Best Seed** - Assign final folds
8. **Validate** - Check fold balance (record counts, PP by fold)
9. **Save Output** - Export to parquet

## Functions in utils.py

| Function | Description |
|----------|-------------|
| `load_and_merge_data()` | Load and merge parquet files |
| `remove_zero_exposure(df)` | Remove records with zero EE |
| `calculate_pure_premium(df)` | Add PP columns |
| `calculate_overall_stats(df)` | Get overall PP and weights |
| `run_simulations(df, stats)` | Test multiple seeds |
| `get_best_seed(results_df)` | Find best seed |
| `plot_objective_distribution(results_df)` | Plot histogram |
| `assign_folds(df, seed)` | Assign folds using a seed |
| `validate_folds(df, stats)` | Print validation statistics |
| `save_output(df, results_df)` | Save output files |

## Configuration (config.py)

Edit `config.py` to change:
- File paths (`MAIN_DATA_PATH`, `SUPERPOLICY_PATH`, `OUTPUT_PATH`)
- Number of simulations (`N_SIMULATIONS = 100`)
- Number of folds (`N_FOLDS = 5`)
- Column mappings (`EE_COLUMNS`, `INCURRED_COLUMNS`)

## Requirements

```
pandas
numpy
matplotlib
pyarrow  # for parquet support
```

## How to Use Output

Join the output back to your original data:

```python
import pandas as pd

# Load original data
original_data = pd.read_parquet("your_data.parquet")

# Load fold assignments
fold_data = pd.read_parquet("/Users/Mach/dev/aps/data/2026_Dmodel_data/fold_superpolicy/car_master_dataset_fold_sp.parquet")

# Merge on vin_date
merged = pd.merge(original_data, fold_data[['vin_date', 'fold']], on='vin_date', how='left')
```
