"""
Configuration file for EE exploration across vehicle types.
"""

# =============================================================================
# VEHICLE TYPES
# =============================================================================

VEHICLE_TYPES = ["car", "truck", "van", "suv"]

# Data path template - {vehicle} will be replaced with car/truck/van/suv
DATA_PATH_TEMPLATE = "/Users/Mach/dev/aps/data/2026_Dmodel_data/master_dataset_{vehicle}.parquet"

# =============================================================================
# COLUMN MAPPINGS
# =============================================================================

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
    "coll": "ee_coll_raw",
    "comp": "ee_comp_raw"
}

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

# =============================================================================
# COLUMN SELECTION
# =============================================================================
# Set to True to use RAW columns, False to use IMPUTED columns
USE_RAW_EE = False
USE_RAW_INCURRED = False

# Active columns (based on selection above)
EE_COLUMNS = EE_COLUMNS_RAW if USE_RAW_EE else EE_COLUMNS_IMPS
INCURRED_COLUMNS = INCURRED_COLUMNS_RAW if USE_RAW_INCURRED else INCURRED_COLUMNS_IMPS

# Coverage names
COVERAGES = ["bi", "pd", "pip", "med", "coll", "comp"]
