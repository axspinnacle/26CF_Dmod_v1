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

# Earned Exposure columns - IMPUTED (6 coverages)
EE_COLUMNS_IMPS = {
    "bi": "ee_bi_imps",
    "pd": "ee_pd_imps",
    "pip": "ee_pip_imps",
    "med": "ee_med_imps",
    "coll": "ee_coll_imp_imps",
    "comp": "ee_comp_imps"
}

# Earned Exposure columns - RAW (6 coverages)
EE_COLUMNS_RAW = {
    "bi": "ee_bi_raw",
    "pd": "ee_pd_raw",
    "pip": "ee_pip_raw",
    "med": "ee_med_raw",
    "coll": "ee_coll_imps",
    "comp": "ee_comp_raw"
}

# =============================================================================
# COLUMN SELECTION
# =============================================================================
# Set to True to use RAW columns, False to use IMPUTED columns
USE_RAW_EE = False

# Active EE columns (based on selection above)
EE_COLUMNS = EE_COLUMNS_RAW if USE_RAW_EE else EE_COLUMNS_IMPS

# Incurred Loss columns - IMPUTED (6 coverages)
INCURRED_COLUMNS_IMPS = {
    "bi": "incurred_raw_bi_imps",
    "pd": "incurred_raw_pd_imps",
    "pip": "incurred_raw_pip_imps",
    "med": "incurred_raw_med_cal_imps",
    "coll": "incurred_raw_coll_imps",
    "comp": "incurred_raw_comp_imps"
}

# Incurred Loss columns - RAW (6 coverages)
INCURRED_COLUMNS_RAW = {
    "bi": "incurred_raw_bi_raw",
    "pd": "incurred_raw_pd_raw",
    "pip": "incurred_raw_pip_raw",
    "med": "incurred_raw_med_raw",
    "coll": "incurred_raw_coll_raw",
    "comp": "incurred_raw_comp_raw"
}

# Set to True to use RAW incurred columns, False to use IMPUTED
USE_RAW_INCURRED = False

# Active INCURRED columns (based on selection above)
INCURRED_COLUMNS = INCURRED_COLUMNS_RAW if USE_RAW_INCURRED else INCURRED_COLUMNS_IMPS

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
