"""
Configuration file for 5-fold data splitting.
Contains file paths and column mappings.
"""

# =============================================================================
# FILE PATHS
# =============================================================================

# Input files
MAIN_DATA_PATH = "/Users/Mach/dev/aps/data/2026_Dmodel_data/master_dataset_car.parquet"
SUPERPOLICY_PATH = "/Users/Mach/dev/aps/data/2026_Dmodel_data/superpolicy/superpolicy_car.parquet"

# Output file
OUTPUT_PATH = "/Users/Mach/dev/aps/data/2026_Dmodel_data/fold_superpolicy/car_master_dataset_fold_sp.parquet"

# Simulation results (CSV for analysis)
SIMULATION_RESULTS_PATH = "/Users/Mach/dev/aps/data/2026_Dmodel_data/fold_superpolicy/simulation_results.csv"

# =============================================================================
# COLUMN MAPPINGS
# =============================================================================

# Join key
JOIN_KEY = "vin_date"
SUPERPOLICY_COL = "superpolicy_id"

# Earned Exposure columns (6 coverages)
EE_COLUMNS = {
    "bi": "ee_bi_imps",
    "pd": "ee_pd_imps",
    "pip": "ee_pip_imps",
    "med": "ee_med_imps",
    "coll": "ee_coll_imp_imps",
    "comp": "ee_comp_imps"
}

# Incurred Loss columns (6 coverages)
INCURRED_COLUMNS = {
    "bi": "incurred_raw_bi_imps",
    "pd": "incurred_raw_pd_imps",
    "pip": "incurred_raw_pip_imps",
    "med": "incurred_raw_med_cal_imps",
    "coll": "incurred_raw_coll_imps",
    "comp": "incurred_raw_comp_imps"
}

# Coverage names
COVERAGES = ["bi", "pd", "pip", "med", "coll", "comp"]

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

N_SIMULATIONS = 100  # Number of random seeds to try
N_FOLDS = 5          # Number of folds for cross-validation

# =============================================================================
# OUTPUT COLUMNS
# =============================================================================

# Pure Premium column names for output
PP_COLUMNS = [f"pp_{cov}" for cov in COVERAGES]

# Final output columns
OUTPUT_COLUMNS = [JOIN_KEY, SUPERPOLICY_COL, "fold"] + PP_COLUMNS
