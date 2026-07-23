"""
encoding_strategies.py
----------------------
Reusable encoding functions for the XGBoost benchmarking pipeline.

Provides four encoding strategies that operate on the same raw training data:

    Type 1 – Ordinal (0-5)  : Keep original 0-5 integer levels unchanged.
                               Non-0-5 features are passed through as-is.
    Type 2 – Binary          : Collapse every 0-5 feature to 0 or 1 using the
                               grouping rule {0,1,2}->0 | {3,4,5}->1.
                               String categoricals are One-Hot Encoded.
    Type 3 – Actuarial       : Per-column strategy from the actuary's docx
                               (LevelMapping.docx, DH-Liab / pp_bi column).
                               Strategies: ordered, binary_low_hi, binary_lo_high,
                               ohe, h_map2to4, ohe_map2to4, group_ohe, drop.
    Type 4 – Custom          : Map 2->4 transformation then binary grouping.
                               For 0-5 features: remap level 2->4, then apply
                               binary {0,1,4}->0 | {3,5}->1. Result: levels
                               {0,1,3,4,5} only (no level 2). String features OHE.

All encoders are fitted strictly on the TRAINING data and then applied to the
test data to prevent leakage.
"""

import os
import json
import ast
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_PATH    = "/Users/Mach/dev/aps/data/2026_Dmodel_data/train_combined.parquet"
TEST_PATH     = "/Users/Mach/dev/aps/data/2026_Dmodel_data/test_combined.parquet"
MAPPING_CSV   = os.path.join(PROJECT_ROOT, "config", "level_mapping_reference.csv")
CONFIG_PATH   = os.path.join(PROJECT_ROOT, "config.json")

# ── Load modeling config from config.json ─────────────────────────────────────
def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)

_cfg              = _load_config()
TARGET            = _cfg["modeling"]["target_column"]           # e.g. "pp_bi"
TARGET_CEILING    = _cfg["modeling"].get("target_ceiling", None)  # e.g. 100000
CEILING_COLS      = set(_cfg["modeling"].get("ceiling_applies_to", []))
EXPOSURE_COL      = _cfg["modeling"].get("exposure_column", None)  # e.g. "ee_bi"

DEBUG_N_TRAIN   = 10_000   # DEBUG=1 : fixed 10K rows
DEBUG_N_TEST    = 2_000    # DEBUG=1 : fixed 2K rows
DEBUG_FRAC_TRAIN = 0.10    # DEBUG=2 : 10% of training data
DEBUG_FRAC_TEST  = 0.10    # DEBUG=2 : 10% of test data

# All potential target columns — always excluded from features
_ALL_TARGETS = {"pp_bi", "pp_pd", "pp_pip", "pp_coll", "pp_comp",
                "ee_bi", "ee_pd", "ee_pip", "ee_coll", "ee_comp"}

# Columns that are identifiers / dates / targets / fold -> always excluded
EXCLUDE_ALWAYS = list(_ALL_TARGETS | {
    "fold", "vin_date", "vin_x", "vin_y",
    "superpolicy_id", "policyid", "reference_num", "companyid",
    "poleffdt", "polexpdt_raw", "polexpdt_imps",
    "coveffdt_raw", "coveffdt_imps", "covexpdt_raw", "covexpdt_imps",
    "origpoleffdt_raw", "origpoleffdt_imps", "poleffdt_imps",
    "zip",
    # Duplicate state column: use st_raw_x (0% NaN) instead
    "insstate",
})


# ============================================================================
# 1. Data Loading
# ============================================================================

def _optimize():
    """Lazy import of optimize_dtypes to avoid circular dependency at module level."""
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(__file__))
    from data_processing import optimize_dtypes
    return optimize_dtypes


def _debug_label(debug: int) -> str:
    return {0: "FULL", 1: "DEBUG-10K", 2: "DEBUG-10%"}.get(debug, "FULL")


def load_train_only(debug: int = 1) -> pd.DataFrame:
    """Return train_df with target rows where TARGET is not null.

    Loads only the training parquet — keeps memory free for encoding/training.
    dtype-optimized immediately after load (float64→float32, int64→int32).

    Parameters
    ----------
    debug : int
        0 = full data
        1 = first 10K rows (fast smoke test)
        2 = random 10% sample (medium-scale run, ~10× bigger than debug=1)
    """
    optimize_dtypes = _optimize()
    label = _debug_label(debug)

    print(f"[{label}] Loading train ...")
    train = pd.read_parquet(TRAIN_PATH)

    if debug == 1:
        train = train.head(DEBUG_N_TRAIN)
    elif debug == 2:
        train = train.sample(frac=DEBUG_FRAC_TRAIN, random_state=42)

    train = optimize_dtypes(train)
    train = train[train[TARGET].notna()].reset_index(drop=True)
    print(f"  Train: {len(train):,} rows")
    return train


def load_test_only(debug: int = 1) -> pd.DataFrame:
    """Return test_df with target rows where TARGET is not null.

    Loads only the test parquet — call this in the separate evaluation notebook
    after all models have been trained and saved.
    dtype-optimized immediately after load (float64→float32, int64→int32).

    Parameters
    ----------
    debug : int
        0 = full data
        1 = first 2K rows
        2 = random 10% sample
    """
    optimize_dtypes = _optimize()
    label = _debug_label(debug)

    print(f"[{label}] Loading test ...")
    test = pd.read_parquet(TEST_PATH)

    if debug == 1:
        test = test.head(DEBUG_N_TEST)
    elif debug == 2:
        test = test.sample(frac=DEBUG_FRAC_TEST, random_state=42)

    test = optimize_dtypes(test)
    test = test[test[TARGET].notna()].reset_index(drop=True)
    print(f"  Test: {len(test):,} rows")
    return test


def load_train_test(debug: int = 1) -> tuple:
    """Return (train_df, test_df) with target rows where TARGET is not null.

    Loads both datasets simultaneously.  Use load_train_only() + load_test_only()
    instead when memory is a concern (full-data runs).

    Parameters
    ----------
    debug : int  – 0 = full, 1 = 10K/2K rows, 2 = 10% sample
    """
    optimize_dtypes = _optimize()
    label = _debug_label(debug)

    print(f"[{label}] Loading train ...")
    train = pd.read_parquet(TRAIN_PATH)
    if debug == 1:
        train = train.head(DEBUG_N_TRAIN)
    elif debug == 2:
        train = train.sample(frac=DEBUG_FRAC_TRAIN, random_state=42)
    train = optimize_dtypes(train)

    print(f"[{label}] Loading test  ...")
    test = pd.read_parquet(TEST_PATH)
    if debug == 1:
        test = test.head(DEBUG_N_TEST)
    elif debug == 2:
        test = test.sample(frac=DEBUG_FRAC_TEST, random_state=42)
    test = optimize_dtypes(test)

    # Drop rows where target is null
    train = train[train[TARGET].notna()].reset_index(drop=True)
    test  = test[test[TARGET].notna()].reset_index(drop=True)

    print(f"  Train: {len(train):,} rows  |  Test: {len(test):,} rows")
    return train, test


def get_y(df: pd.DataFrame) -> pd.Series:
    """
    Extract target column and apply ceiling cap if configured.

    Ceiling (from config.json -> modeling.target_ceiling) is only applied
    to columns listed in modeling.ceiling_applies_to (e.g. pp_bi, pp_pd).

    To switch targets, change 'target_column' in config.json.
    """
    y = df[TARGET].astype("float32")
    if TARGET_CEILING is not None and TARGET in CEILING_COLS:
        n_capped = (y > TARGET_CEILING).sum()
        y = y.clip(upper=float(TARGET_CEILING))
        if n_capped > 0:
            print(f"  [Ceiling] Applied {TARGET_CEILING:,} cap to '{TARGET}': "
                  f"{n_capped:,} rows capped ({100*n_capped/len(y):.2f}%)")
    return y


# ============================================================================
# 2. Feature catalogue helpers
# ============================================================================

def _load_mapping() -> pd.DataFrame:
    """Load the level mapping reference CSV."""
    return pd.read_csv(MAPPING_CSV)


def _get_numeric_features(df: pd.DataFrame) -> list:
    """All numeric columns except excluded ones."""
    return [c for c in df.select_dtypes(include=[np.number]).columns
            if c not in EXCLUDE_ALWAYS]


def _get_object_features(df: pd.DataFrame) -> list:
    """All object/string columns except excluded ones."""
    return [c for c in df.select_dtypes(include="object").columns
            if c not in EXCLUDE_ALWAYS]


def _is_0_5_col(s: pd.Series) -> bool:
    """True if the numeric column has integer-like values 0-5 only."""
    if not pd.api.types.is_numeric_dtype(s):
        return False
    vals = s.dropna().unique()
    if len(vals) == 0:          # all-NaN column → not a 0-5 col
        return False
    return (len(vals) <= 7) and (float(vals.min()) >= 0) and (float(vals.max()) <= 5)


# ============================================================================
# 3. Low-level encoders
# ============================================================================

def _apply_binary_map(s: pd.Series, low_set: set, high_set: set,
                      fill_unknown: int = 0) -> pd.Series:
    """Map low_set values -> 0, high_set values -> 1, unknown -> fill_unknown."""
    def _mapper(v):
        if pd.isna(v):
            return np.nan
        v = int(round(v))
        if v in low_set:
            return 0
        if v in high_set:
            return 1
        return fill_unknown
    return s.map(_mapper).astype("float32")


def _apply_h_map2to4(s: pd.Series) -> pd.Series:
    """Hierarchical: remap level 2 -> 4, keep other levels unchanged."""
    return s.map(lambda v: 4 if (not pd.isna(v) and int(round(v)) == 2) else v).astype("float32")


def _fit_ohe_for_col(col: str, train_vals: pd.Series) -> OneHotEncoder:
    """Fit a OneHotEncoder for a single column."""
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.float32)
    enc.fit(train_vals.dropna().values.reshape(-1, 1))
    return enc


def _transform_ohe(col: str, s: pd.Series, enc: OneHotEncoder) -> pd.DataFrame:
    """Apply a fitted OHE to a Series. Returns a DataFrame with named columns."""
    arr = s.fillna("__unknown__").values.reshape(-1, 1)
    ohe_arr = enc.transform(arr)
    cat_names = [str(c) for c in enc.categories_[0]]
    cols = [f"{col}__{c}" for c in cat_names]
    return pd.DataFrame(ohe_arr, columns=cols, index=s.index)


# ============================================================================
# 4. Encoding strategy functions
# ============================================================================

def encode_type1_ordinal(train: pd.DataFrame, test: pd.DataFrame = None):
    """
    Type 1 – Ordinal: pass all numeric features through unchanged.
    Object columns are One-Hot Encoded (fit on train, applied to test).

    Parameters
    ----------
    train : DataFrame  – training data (used to fit encoders)
    test  : DataFrame or None – if None, only X_train + encoders are returned
            (use encode_type1_ordinal_transform to apply to test later)

    Returns
    -------
    If test is provided : X_train, X_test, feature_names, encoders
    If test is None     : X_train, feature_names, encoders
    """
    print("  [Type 1] Ordinal encoding ...")
    num_cols = _get_numeric_features(train)
    obj_cols = _get_object_features(train)

    # Numeric columns: keep ordinal (no change)
    X_tr = train[num_cols].astype("float32").copy()
    feat_names = list(X_tr.columns)

    # String columns: One-Hot Encode (fit on train)
    ohe_parts_tr, ohe_names = [], []
    encoders = {}
    for col in obj_cols:
        enc = _fit_ohe_for_col(col, train[col].astype(str))
        encoders[col] = enc
        ohe_tr = _transform_ohe(col, train[col].astype(str), enc)
        ohe_parts_tr.append(ohe_tr)
        ohe_names.extend(ohe_tr.columns.tolist())

    if ohe_parts_tr:
        X_tr = pd.concat([X_tr] + ohe_parts_tr, axis=1)
        feat_names.extend(ohe_names)

    print(f"     Features: {len(feat_names)}  (numeric ordinal: {len(num_cols)}, OHE string: {len(ohe_names)})")

    if test is None:
        return X_tr, feat_names, encoders

    # Apply to test
    X_te = test[num_cols].astype("float32").copy()
    ohe_parts_te = []
    for col in obj_cols:
        enc = encoders[col]
        ohe_te = _transform_ohe(col, test[col].astype(str), enc)
        ohe_parts_te.append(ohe_te)
    if ohe_parts_te:
        X_te = pd.concat([X_te] + ohe_parts_te, axis=1)

    return X_tr, X_te, feat_names, encoders


def encode_type2_binary(train: pd.DataFrame, test: pd.DataFrame = None):
    """
    Type 2 – Binary: collapse every 0-5 feature to {0,1} using default
    {0,1,2}->0, {3,4,5}->1. Non-0-5 numerics pass through. Object columns
    are One-Hot Encoded (fit on train).

    Parameters
    ----------
    train : DataFrame  – training data (used to fit encoders)
    test  : DataFrame or None – if None, only X_train + encoders are returned

    Returns
    -------
    If test is provided : X_train, X_test, feature_names, encoders
    If test is None     : X_train, feature_names, encoders
    """
    print("  [Type 2] Binary encoding ...")
    num_cols = _get_numeric_features(train)
    obj_cols = _get_object_features(train)

    low_set  = {0, 1, 2}
    high_set = {3, 4, 5}

    X_tr = pd.DataFrame(index=train.index)

    encoders = {}
    for col in num_cols:
        if _is_0_5_col(train[col]):
            X_tr[col] = _apply_binary_map(train[col], low_set, high_set)
        else:
            X_tr[col] = train[col].astype("float32")

    # String columns: One-Hot Encode (fit on train)
    ohe_parts_tr, ohe_names = [], []
    for col in obj_cols:
        enc = _fit_ohe_for_col(col, train[col].astype(str))
        encoders[col] = enc
        ohe_tr = _transform_ohe(col, train[col].astype(str), enc)
        ohe_parts_tr.append(ohe_tr)
        ohe_names.extend(ohe_tr.columns.tolist())

    if ohe_parts_tr:
        X_tr = pd.concat([X_tr] + ohe_parts_tr, axis=1)

    feat_names = list(X_tr.columns)
    n_binary = sum(_is_0_5_col(train[c]) for c in num_cols)
    print(f"     Features: {len(feat_names)}  (0-5 binary: {n_binary}, other numeric: {len(num_cols)-n_binary}, OHE string: {len(ohe_names)})")

    if test is None:
        return X_tr, feat_names, encoders

    # Apply to test
    X_te = pd.DataFrame(index=test.index)
    for col in num_cols:
        if _is_0_5_col(train[col]):
            X_te[col] = _apply_binary_map(test[col], low_set, high_set)
        else:
            X_te[col] = test[col].astype("float32")
    ohe_parts_te = []
    for col in obj_cols:
        ohe_te = _transform_ohe(col, test[col].astype(str), encoders[col])
        ohe_parts_te.append(ohe_te)
    if ohe_parts_te:
        X_te = pd.concat([X_te] + ohe_parts_te, axis=1)

    return X_tr, X_te, feat_names, encoders


def encode_type3_actuarial(train: pd.DataFrame, test: pd.DataFrame = None):
    """
    Type 3 – Actuarial: per-column strategy from LevelMapping.docx (DH-Liab).

    Strategies applied:
      ordered       -> keep 0-5 numeric as-is
      binary_low_hi -> {0,1,2}->0, {3,4,5}->1
      binary_lo_high-> {0,1,2,3}->0, {4,5}->1
      ohe           -> OneHotEncoder (fit on train)
      h_map2to4     -> remap 2->4, keep ordinal
      ohe_map2to4   -> remap 2->4 then OHE
      group_ohe     -> OHE (string grouping delegated to OHE's handle_unknown)
      drop          -> feature excluded
      not_specified -> pass through numeric; label-encode strings

    Parameters
    ----------
    train : DataFrame – training data (encoders fitted here)
    test  : DataFrame or None – if None returns (X_train, feat_names, encoders)

    Returns
    -------
    If test is provided : X_train, X_test, feat_names, encoders
    If test is None     : X_train, feat_names, encoders
    """
    print("  [Type 3] Actuarial encoding ...")
    mapping = _load_mapping()
    strategy_map = dict(zip(mapping["feature"], mapping["type_3_strategy_code"]))
    binary_groups_map = dict(zip(mapping["feature"], mapping["type_3_binary_groups"]))

    num_cols = _get_numeric_features(train)
    obj_cols = _get_object_features(train)
    all_cols = num_cols + obj_cols

    X_tr_parts = []
    feat_names = []
    encoders   = {}

    LOW_HI  = ({0,1,2}, {3,4,5})
    LO_HIGH = ({0,1,2,3}, {4,5})

    for col in all_cols:
        strat = strategy_map.get(col, "not_specified")

        if strat == "drop":
            continue

        elif strat in ("ordered", "not_specified"):
            if col in num_cols:
                X_tr_parts.append(train[[col]].astype("float32"))
                feat_names.append(col)
            else:
                cat = pd.Categorical(train[col])
                tr_s = pd.Series(cat.codes, index=train.index, name=col, dtype="float32")
                X_tr_parts.append(tr_s.to_frame())
                feat_names.append(col)
                encoders[col] = ("label", cat.categories)

        elif strat == "binary_low_hi":
            s_tr = _apply_binary_map(train[col], *LOW_HI)
            X_tr_parts.append(s_tr.rename(col).to_frame())
            feat_names.append(col)

        elif strat == "binary_lo_high":
            s_tr = _apply_binary_map(train[col], *LO_HIGH)
            X_tr_parts.append(s_tr.rename(col).to_frame())
            feat_names.append(col)

        elif strat in ("ohe", "group_ohe"):
            enc = _fit_ohe_for_col(col, train[col].astype(str))
            ohe_tr = _transform_ohe(col, train[col].astype(str), enc)
            X_tr_parts.append(ohe_tr)
            feat_names.extend(ohe_tr.columns.tolist())
            encoders[col] = ("ohe", enc)

        elif strat == "h_map2to4":
            s_tr = _apply_h_map2to4(train[col])
            X_tr_parts.append(s_tr.rename(col).to_frame())
            feat_names.append(col)

        elif strat == "ohe_map2to4":
            s_tr = _apply_h_map2to4(train[col]).astype(str)
            enc = _fit_ohe_for_col(col, s_tr)
            ohe_tr = _transform_ohe(col, s_tr, enc)
            X_tr_parts.append(ohe_tr)
            feat_names.extend(ohe_tr.columns.tolist())
            encoders[col] = ("ohe_map2to4", enc)

    # Also store strategy_map in encoders for transform step
    encoders["__strategy_map__"] = strategy_map
    encoders["__num_cols__"]     = num_cols
    encoders["__obj_cols__"]     = obj_cols

    X_train = pd.concat(X_tr_parts, axis=1)
    print(f"     Features: {len(feat_names)}")

    if test is None:
        return X_train, feat_names, encoders

    # Apply to test using fitted encoders
    X_te_parts = []
    for col in all_cols:
        strat = strategy_map.get(col, "not_specified")
        if strat == "drop":
            continue
        elif strat in ("ordered", "not_specified"):
            if col in num_cols:
                X_te_parts.append(test[[col]].astype("float32"))
            else:
                cats = encoders[col][1]
                te_s = pd.Series(pd.Categorical(test[col], categories=cats).codes,
                                 index=test.index, name=col, dtype="float32")
                X_te_parts.append(te_s.to_frame())
        elif strat == "binary_low_hi":
            X_te_parts.append(_apply_binary_map(test[col], *LOW_HI).rename(col).to_frame())
        elif strat == "binary_lo_high":
            X_te_parts.append(_apply_binary_map(test[col], *LO_HIGH).rename(col).to_frame())
        elif strat in ("ohe", "group_ohe"):
            enc = encoders[col][1]
            X_te_parts.append(_transform_ohe(col, test[col].astype(str), enc))
        elif strat == "h_map2to4":
            X_te_parts.append(_apply_h_map2to4(test[col]).rename(col).to_frame())
        elif strat == "ohe_map2to4":
            enc = encoders[col][1]
            s_te = _apply_h_map2to4(test[col]).astype(str)
            X_te_parts.append(_transform_ohe(col, s_te, enc))

    X_test = pd.concat(X_te_parts, axis=1)
    return X_train, X_test, feat_names, encoders


def encode_type4_custom(train: pd.DataFrame, test: pd.DataFrame = None):
    """
    Type 4 – Custom: Map 2->4 transformation + binary grouping.
    
    For all 0-5 features:
      1. Remap level 2 -> level 4  (using _apply_h_map2to4)
      2. Apply binary grouping: {0,1,4}->0 | {3,5}->1
      
    Result: Features will only have levels {0, 1, 3, 4, 5} (no level 2).
    Non-0-5 numeric features pass through unchanged.
    String features are One-Hot Encoded (fit on train).

    Parameters
    ----------
    train : DataFrame – training data (encoders fitted here)
    test  : DataFrame or None – if None returns (X_train, feat_names, encoders)

    Returns
    -------
    If test is provided : X_train, X_test, feat_names, encoders
    If test is None     : X_train, feat_names, encoders
    """
    print("  [Type 4] Custom encoding (map 2->4, then binary {0,1,4}->0 | {3,5}->1) ...")
    num_cols = _get_numeric_features(train)
    obj_cols = _get_object_features(train)

    # Define binary groups AFTER 2->4 remapping
    # After remapping: 0,1,4 are "low", 3,5 are "high"
    low_set  = {0, 1, 4}
    high_set = {3, 5}

    X_tr = pd.DataFrame(index=train.index)
    encoders = {}
    
    # Process numeric columns
    for col in num_cols:
        if _is_0_5_col(train[col]):
            # Step 1: Remap 2->4
            remapped = _apply_h_map2to4(train[col])
            # Step 2: Apply binary grouping
            X_tr[col] = _apply_binary_map(remapped, low_set, high_set)
        else:
            # Non-0-5 features: pass through
            X_tr[col] = train[col].astype("float32")

    # String columns: One-Hot Encode (fit on train)
    ohe_parts_tr, ohe_names = [], []
    for col in obj_cols:
        enc = _fit_ohe_for_col(col, train[col].astype(str))
        encoders[col] = enc
        ohe_tr = _transform_ohe(col, train[col].astype(str), enc)
        ohe_parts_tr.append(ohe_tr)
        ohe_names.extend(ohe_tr.columns.tolist())

    if ohe_parts_tr:
        X_tr = pd.concat([X_tr] + ohe_parts_tr, axis=1)

    feat_names = list(X_tr.columns)
    n_mapped = sum(_is_0_5_col(train[c]) for c in num_cols)
    print(f"     Features: {len(feat_names)}  (0-5 map2to4+binary: {n_mapped}, other numeric: {len(num_cols)-n_mapped}, OHE string: {len(ohe_names)})")

    if test is None:
        return X_tr, feat_names, encoders

    # Apply to test
    X_te = pd.DataFrame(index=test.index)
    for col in num_cols:
        if _is_0_5_col(train[col]):  # Use train to determine if 0-5
            remapped = _apply_h_map2to4(test[col])
            X_te[col] = _apply_binary_map(remapped, low_set, high_set)
        else:
            X_te[col] = test[col].astype("float32")
    
    ohe_parts_te = []
    for col in obj_cols:
        ohe_te = _transform_ohe(col, test[col].astype(str), encoders[col])
        ohe_parts_te.append(ohe_te)
    if ohe_parts_te:
        X_te = pd.concat([X_te] + ohe_parts_te, axis=1)

    return X_tr, X_te, feat_names, encoders
