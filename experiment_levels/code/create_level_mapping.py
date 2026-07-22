"""
create_level_mapping.py
-----------------------
Program 1 of 2 in the XGBoost benchmarking pipeline.

Purpose:
    Examine the training dataset's categorical features and produce a
    comprehensive CSV reference file (config/level_mapping_reference.csv)
    that documents four alternative encoding strategies for each feature:

    Type 1 – Ordinal      : Keep original 0-5 integer levels unchanged.
    Type 2 – Binary       : Collapse levels into 0 / 1 using a low/high threshold.
    Type 3 – Actuarial    : Expert encoding per actuary's LevelMapping.docx
                            (tuned for the Liability / pp_bi coverage).
    Type 4 – Custom       : Placeholder column – fill in manually later.

Usage:
    1. Set DEBUG = 1 (default) to process only 10 000 rows for a fast run.
    2. Run: python code/create_level_mapping.py
    3. Review / edit  config/level_mapping_reference.csv as needed.
    4. Set DEBUG = 0 and re-run for a full-data cardinality scan (optional).

Output:
    config/level_mapping_reference.csv
"""

# ── Debug flag ────────────────────────────────────────────────────────────────
DEBUG = 1          # 1 = load 10 000 rows;  0 = load full dataset
DEBUG_ROWS = 10_000
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_PATH    = "/Users/Mach/dev/aps/data/2026_Dmodel_data/train_combined.parquet"
OUTPUT_DIR    = os.path.join(PROJECT_ROOT, "config")
OUTPUT_CSV    = os.path.join(OUTPUT_DIR, "level_mapping_reference.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# 1. Actuarial mapping table (from LevelMapping.docx – DH Liab / pp_bi column)
#
# Strategy codes used in this table:
#   "ordered"          -> Type 3 = keep ordinal 0-5
#   "binary_low_hi"    -> Type 3 = {0,1,2}->0, {3,4,5}->1
#   "binary_lo_high"   -> Type 3 = {0,1,2,3}->0, {4,5}->1
#   "ohe"              -> Type 3 = One-Hot Encoding
#   "h_map2to4"        -> Type 3 = remap level 2->4, then keep ordinal
#   "ohe_map2to4"      -> Type 3 = remap level 2->4, then One-Hot
#   "group_ohe"        -> Type 3 = group raw strings, then One-Hot
#   "drop"             -> Type 3 = exclude feature from model
#   "na"               -> Not applicable / not in this coverage
# ============================================================================

ACTUARIAL_LIAB = {
    # ── 0-5 level vc_* features ──────────────────────────────────────────────
    "vc_trailer_assist_raw":                        "ordered",
    "vc_tactile_forward_collision_warning_raw":     "ordered",
    "vc_wipers_speed_sensitive_raw":                "binary_low_hi",      # {0,1,2}->0, {3,4,5}->1
    "vc_wipers_heated_raw":                         "binary_lo_high",     # {0,1,2,3}->0, {4,5}->1
    "vc_wipers_automatic_raw":                      "binary_lo_high",     # {0,1,2,3}->0, {4,5}->1
    "vc_traffic_sign_recognition_raw":              "binary_lo_high",     # {0,1,2,3}->0, {4,5}->1
    "vc_off_road_lights_raw":                       "ordered",
    "vc_power_folding_side_mirrors_raw":            "binary_low_hi",      # {0,1,2}->0, {3,4,5}->1
    "vc_post_crash_fuel_cutoff_raw":                "binary_lo_high",     # {0,1,2,3}->0, {4,5}->1
    "vc_mild_hybrid_raw":                           "ordered",
    "vc_rear_window_defogger_raw":                  "ordered",
    "vc_reverse_automatic_emergency_braking_raw":   "binary_low_hi",      # {0,1,2}->0, {3,4,5}->1
    "vc_video_display_raw":                         "drop",
    "vc_video_system_raw":                          "binary_low_hi",      # {0,1,2}->0, {3,4,5}->1
    "vc_semiautomatic_parking_system_raw":          "binary_lo_high",     # {0,1,2,3}->0, {4,5}->1
    "vc_automatic_emergency_steering_raw":          "ordered",
    "vc_automatic_high_beams_raw":                  "binary_lo_high",     # {0,1,2,3}->0, {4,5}->1
    "vc_auto_dimming_side_mirrors_raw":             "ordered",
    "vc_auto_hazard_flashers_raw":                  "binary_low_hi",      # {0,1,2}->0, {3,4,5}->1
    "vc_auto_start_stop_raw":                       "ohe",
    "vc_automatic_emergency_braking_raw":           "ohe",
    "vc_crash_sensing_door_locks_raw":              "binary_low_hi",      # {0,1,2}->0, {3,4,5}->1
    "vc_active_parking_assistance_raw":             "ohe",
    "vc_adaptive_cruise_control_raw":               "ohe",
    "vc_camera_front_raw":                          "ordered",
    "vc_center_locking_differential_raw":           "binary_low_hi",      # {0,1,2}->0, {3,4,5}->1
    "vc_camera_washer_raw":                         "ohe",
    "vc_camera_side_raw":                           "ordered",
    "vc_cornering_lights_raw":                      "ohe",
    "vc_cornering_brake_control_raw":               "ordered",
    "vc_active_driving_assistance_raw":             "ohe",
    "vc_pedestrian_detection_raw":                  "ohe",
    "vc_integrated_wifi_raw":                       "ordered",
    "vc_lane_keeping_assistance_raw":               "binary_lo_high",     # {0,1,2,3}->0, {4,5}->1 (approx)
    "vc_headlights_wiper_activated_raw":            "ordered",
    "vc_driver_attention_system_raw":               "ohe",
    "vc_glass_roof_raw":                            "ordered",
    "vc_forward_automatic_emergency_braking_raw":   "ohe",                # OHT in docx
    "vc_fog_lights_rear_raw":                       "ohe",
    "vc_electronic_4wd_selector_raw":               "ordered",
    "vc_headlights_leveling_raw":                   "ohe",
    "vc_heated_steering_wheel_raw":                 "binary_lo_high",     # {0,1,2,3}->0, {4,5}->1
    "vc_blind_spot_camera_raw":                     "binary_low_hi",      # {0,1,2}->0, {3,4,5}->1
    "vc_heated_washer_jets_raw":                    "ordered",
    "vc_vehicle_immobilizer_raw":                   "binary_low_hi",      # {0,1,2}->0, {3,4,5}->1
    "vc_tinted_windows_raw":                        "ohe",
    "vc_speed_sensitive_volume_raw":                "binary_low_hi",      # {0,1,2}->0, {3,4,5}->1
    "vc_sunroof_raw":                               "ohe",
    "vc_variable_power_steering_raw":               "binary_lo_high",     # {0,1,2,3}->0, {4,5}->1
    "vc_parking_sensors_raw":                       "binary_low_hi",      # {0,1,2}->0, {3,4,5}->1
    "vc_parking_collision_warning_raw":             "binary_low_hi",      # {0,1,2}->0, {3,4,5}->1
    "vc_passive_collision_avoidance_system_raw":    "ohe",
    "vc_precollision_system_raw":                   "ordered",
    "vc_power_sunroof_raw":                         "ohe",
    "vc_remote_engine_start_raw":                   "ohe",
    "vc_rear_wiper_raw":                            "ohe",
    "vc_rear_parking_sensors_raw":                  "binary_low_hi",      # {0,1,2}->0, {3,4,5}->1
    "vc_rear_cross_traffic_warning_raw":            "binary_lo_high",     # {0,1,2,3}->0, {4,5}->1
    "vc_visual_forward_collision_warning_raw":      "ohe",
    "vc_side_mirror_turn_signals_raw":              "binary_low_hi",      # {0,1,2}->0, {3,4,5}->1
    "vc_navigation_touchscreen_raw":                "binary_lo_high",     # {0,1,2,3}->0, {4,5}->1
    "vc_backup_camera_raw":                         "ohe",
    "vc_auto_dimming_rearview_mirror_raw":          "ohe",
    "vc_audible_forward_collision_warning_raw":     "ohe",
    "vc_cross_traffic_warning_raw":                 "binary_low_hi",      # {0,1,2}->0, {3,4,5}->1
    "vc_crumple_zones_raw":                         "binary_low_hi",      # {0,1,2}->0, {3,4,5}->1
    "vc_blind_spot_warning_raw":                    "binary_lo_high",     # {0,1,2,3}->0, {4,5}->1
    "vc_crash_sensors_raw":                         "ohe",
    "vc_active_collision_avoidance_system_raw":     "ohe",
    "vc_lane_departure_warning_raw":                "binary_lo_high",     # {0,1,2,3}->0, {4,5}->1
    "vc_daytime_running_lights_raw":                "drop",
    "vc_headlights_auto_on_off_raw":                "binary_low_hi",      # {0,1,2}->0, {3,4,5}->1
    "vc_front_parking_sensors_raw":                 "binary_low_hi",      # {0,1,2}->0, {3,4,5}->1
    "vc_forward_collision_warning_raw":             "ohe",
    "vc_fog_lights_raw":                            "drop",
    "vc_electronic_braking_system_raw":             "ohe",
    "vc_heated_side_mirrors_raw":                   "binary_low_hi",      # {0,1,2}->0, {3,4,5}->1
    "vc_hill_ascent_assist_raw":                    "ohe",
    "vc_camera_raw":                                "ohe",
    "vc_instrumentation_display_cal":               "ordered",
    "vc_infotainment_screen_cal":                   "ordered",
    "vc_side_window_defogger_raw":                  "drop",
    "vc_antitheft_key_raw":                         "drop",
    "vc_front_cross_traffic_warning_raw":           "ordered",
    # ── H (Map 2 to 4) features ───────────────────────────────────────────────
    "vc_subwoofer_raw":                             "h_map2to4",
    "vc_premium_audio_raw":                         "h_map2to4",
    "vc_satellite_radio_raw":                       "h_map2to4",
    "vc_taillights_led_raw":                        "h_map2to4",
    "vc_dvd_player_raw":                            "h_map2to4",
    "vc_surround_sound_raw":                        "ohe_map2to4",        # OHE (Map 2 to 4)
    "vc_touch_screen_raw":                          "ohe_map2to4",        # OHE (Map 2 to 4)
    "vc_softtop_raw":                               "ohe_map2to4",        # OHE (Map 2 to 4)
    # ── One-Hot / aspiration ──────────────────────────────────────────────────
    "vc_aspiration_raw":                            "ohe",
    # ── Group and One-Hot (high-cardinality string categoricals) ─────────────
    "vehicle_use_raw":                              "group_ohe",
    "owt_current_ownership_type_raw":               "group_ohe",
    "vc_fuel_type_raw":                             "group_ohe",
    "dml_body_style_raw":                           "group_ohe",
    "st_raw_x":                                     "group_ohe",
    "insstate":                                     "drop",           # duplicate of st_raw_x; 0.61% NaN vs 0%
    "dml_make_raw":                                 "group_ohe",
}

# ── Binary grouping rules for each applicable feature ─────────────────────────
# Key = column name, Value = tuple (low_group, high_group) as frozensets
BINARY_GROUPS = {
    # low/high pattern  {0,1,2} -> 0 | {3,4,5} -> 1
    "vc_trailer_assist_raw":                        ({0,1,2}, {3,4,5}),
    "vc_wipers_speed_sensitive_raw":                ({0,1,2}, {3,4,5}),
    "vc_power_folding_side_mirrors_raw":            ({0,1,2}, {3,4,5}),
    "vc_reverse_automatic_emergency_braking_raw":   ({0,1,2}, {3,4,5}),
    "vc_video_system_raw":                          ({0,1,2}, {3,4,5}),
    "vc_crash_sensing_door_locks_raw":              ({0,1,2}, {3,4,5}),
    "vc_camera_front_raw":                          ({0,1,2}, {3,4,5}),
    "vc_center_locking_differential_raw":           ({0,1,2}, {3,4,5}),
    "vc_auto_hazard_flashers_raw":                  ({0,1,2}, {3,4,5}),
    "vc_blind_spot_camera_raw":                     ({0,1,2}, {3,4,5}),
    "vc_vehicle_immobilizer_raw":                   ({0,1,2}, {3,4,5}),
    "vc_speed_sensitive_volume_raw":                ({0,1,2}, {3,4,5}),
    "vc_parking_sensors_raw":                       ({0,1,2}, {3,4,5}),
    "vc_parking_collision_warning_raw":             ({0,1,2}, {3,4,5}),
    "vc_rear_parking_sensors_raw":                  ({0,1,2}, {3,4,5}),
    "vc_side_mirror_turn_signals_raw":              ({0,1,2}, {3,4,5}),
    "vc_cross_traffic_warning_raw":                 ({0,1,2}, {3,4,5}),
    "vc_crumple_zones_raw":                         ({0,1,2}, {3,4,5}),
    "vc_headlights_auto_on_off_raw":                ({0,1,2}, {3,4,5}),
    "vc_front_parking_sensors_raw":                 ({0,1,2}, {3,4,5}),
    "vc_heated_side_mirrors_raw":                   ({0,1,2}, {3,4,5}),
    "vc_visual_forward_collision_warning_raw":      ({0,1,2}, {3,4,5}),
    "vc_audible_forward_collision_warning_raw":     ({0,1,2}, {3,4,5}),
    # lo/high pattern  {0,1,2,3} -> 0 | {4,5} -> 1
    "vc_wipers_heated_raw":                         ({0,1,2,3}, {4,5}),
    "vc_wipers_automatic_raw":                      ({0,1,2,3}, {4,5}),
    "vc_traffic_sign_recognition_raw":              ({0,1,2,3}, {4,5}),
    "vc_post_crash_fuel_cutoff_raw":                ({0,1,2,3}, {4,5}),
    "vc_semiautomatic_parking_system_raw":          ({0,1,2,3}, {4,5}),
    "vc_automatic_high_beams_raw":                  ({0,1,2,3}, {4,5}),
    "vc_lane_keeping_assistance_raw":               ({0,1,2,3}, {4,5}),
    "vc_heated_steering_wheel_raw":                 ({0,1,2,3}, {4,5}),
    "vc_variable_power_steering_raw":               ({0,1,2,3}, {4,5}),
    "vc_rear_cross_traffic_warning_raw":            ({0,1,2,3}, {4,5}),
    "vc_navigation_touchscreen_raw":                ({0,1,2,3}, {4,5}),
    "vc_blind_spot_warning_raw":                    ({0,1,2,3}, {4,5}),
    "vc_lane_departure_warning_raw":                ({0,1,2,3}, {4,5}),
}


# ============================================================================
# 2. Load data
# ============================================================================

def load_data(path: str, debug: bool = True, n_rows: int = DEBUG_ROWS) -> pd.DataFrame:
    if debug:
        print(f"[DEBUG mode] Loading first {n_rows:,} rows from:\n  {path}")
        df = pd.read_parquet(path)
        df = df.head(n_rows)
    else:
        print(f"[FULL mode] Loading full dataset from:\n  {path}")
        df = pd.read_parquet(path)
    print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    return df


# ============================================================================
# 3. Build mapping reference rows
# ============================================================================

def _fmt_vals(vals: list, n: int = 10) -> str:
    """Truncate list to n items and format as string."""
    return str(sorted(vals)[:n])


def _ordinal_map_str() -> str:
    """Type 1: keep levels as-is (identity mapping)."""
    return "0->0, 1->1, 2->2, 3->3, 4->4, 5->5"


def _binary_default_map_str() -> str:
    """Type 2 default: {0,1,2}->0, {3,4,5}->1."""
    return "{0,1,2}->0 | {3,4,5}->1"


def _actuarial_map_str(col: str) -> str:
    """Return a human-readable description of the Type 3 actuarial rule."""
    strategy = ACTUARIAL_LIAB.get(col, "not_specified")
    rules = {
        "ordered":       "ordinal_0_5",
        "binary_low_hi": "binary: {0,1,2}->0 | {3,4,5}->1",
        "binary_lo_high":"binary: {0,1,2,3}->0 | {4,5}->1",
        "ohe":           "one_hot_encode",
        "h_map2to4":     "remap_2to4_then_ordinal",
        "ohe_map2to4":   "remap_2to4_then_ohe",
        "group_ohe":     "group_then_ohe",
        "drop":          "DROP",
        "na":            "N/A",
        "not_specified": "not_in_docx",
    }
    return rules.get(strategy, strategy)


def build_mapping_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a comprehensive mapping reference table from the DataFrame.

    For every column in ACTUARIAL_LIAB plus any remaining vc_*_raw columns,
    produce one row documenting:
      - feature metadata (dtype, cardinality, value range)
      - Type 1 encoding description
      - Type 2 encoding description
      - Type 3 actuarial encoding description (pp_bi / Liability)
      - Type 4 custom (empty placeholder)
    """
    records = []

    # Union of actuarial-specified columns + any vc_*_raw in the data
    vc_in_data = [c for c in df.columns if c.startswith("vc_") and c.endswith("_raw")]
    cal_in_data = [c for c in df.columns if c.startswith("vc_") and c.endswith("_cal")]
    all_cols = list(dict.fromkeys(
        list(ACTUARIAL_LIAB.keys()) + vc_in_data + cal_in_data
    ))

    for col in all_cols:
        in_data = col in df.columns
        if not in_data:
            records.append({
                "feature":               col,
                "in_training_data":      False,
                "dtype":                 "N/A",
                "n_unique":              0,
                "min_val":               "",
                "max_val":               "",
                "is_0_5_range":          False,
                "sample_values":         "NOT IN DATASET",
                "type_1_ordinal":        "N/A",
                "type_1_notes":          "",
                "type_2_binary":         "N/A",
                "type_2_grouping":       "",
                "type_3_actuarial_liab": "N/A",
                "type_3_strategy_code":  "",
                "type_3_binary_groups":  "",
                "type_4_custom":         "",
                "type_4_notes":          "",
            })
            continue

        s          = df[col]
        dtype      = str(s.dtype)
        n_unique   = int(s.nunique())
        is_numeric = pd.api.types.is_numeric_dtype(s)

        if is_numeric:
            mn = float(s.min()) if not s.isna().all() else None
            mx = float(s.max()) if not s.isna().all() else None
            vals = sorted(s.dropna().unique().tolist())[:12]
            is_0_5 = (mn is not None) and (mn >= 0) and (mx is not None) and (mx <= 5) and (n_unique <= 7)
        else:
            mn, mx = None, None
            vals = s.dropna().unique().tolist()[:12]
            is_0_5 = False

        strategy = ACTUARIAL_LIAB.get(col, "not_specified")

        # ── Type 2 binary grouping rule ──────────────────────────────────────
        bin_group  = BINARY_GROUPS.get(col)
        if bin_group:
            low_g, hi_g = bin_group
            t2_grouping = f"{sorted(low_g)}->0 | {sorted(hi_g)}->1"
        elif is_0_5:
            t2_grouping = "{0,1,2}->0 | {3,4,5}->1 (default)"
        else:
            t2_grouping = "N/A"

        # ── Type 3 actuarial rule ─────────────────────────────────────────────
        t3_bin_groups = ""
        if strategy in ("binary_low_hi",):
            bg = BINARY_GROUPS.get(col, ({0,1,2}, {3,4,5}))
            t3_bin_groups = f"{sorted(bg[0])}->0 | {sorted(bg[1])}->1"
        elif strategy in ("binary_lo_high",):
            bg = BINARY_GROUPS.get(col, ({0,1,2,3}, {4,5}))
            t3_bin_groups = f"{sorted(bg[0])}->0 | {sorted(bg[1])}->1"

        records.append({
            "feature":               col,
            "in_training_data":      True,
            "dtype":                 dtype,
            "n_unique":              n_unique,
            "min_val":               str(mn) if mn is not None else "",
            "max_val":               str(mx) if mx is not None else "",
            "is_0_5_range":          is_0_5,
            "sample_values":         _fmt_vals(vals),
            "type_1_ordinal":        _ordinal_map_str() if is_0_5 else "keep as-is (non 0-5)",
            "type_1_notes":          "",
            "type_2_binary":         _binary_default_map_str() if is_0_5 else "N/A",
            "type_2_grouping":       t2_grouping,
            "type_3_actuarial_liab": _actuarial_map_str(col),
            "type_3_strategy_code":  strategy,
            "type_3_binary_groups":  t3_bin_groups,
            "type_4_custom":         "",   # ← fill in manually later
            "type_4_notes":          "",   # ← fill in manually later
        })

    return pd.DataFrame(records)


# ============================================================================
# 4. Main
# ============================================================================

def main():
    print("=" * 65)
    print("  PROGRAM 1 — Create Level Mapping Reference CSV")
    print(f"  DEBUG = {DEBUG}  |  rows = {'10 000' if DEBUG else 'full'}")
    print("=" * 65)
    print()

    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_data(TRAIN_PATH, debug=bool(DEBUG), n_rows=DEBUG_ROWS)

    # ── Build table ───────────────────────────────────────────────────────────
    print("Building mapping reference table ...")
    mapping_df = build_mapping_table(df)
    print(f"  Rows in mapping table : {len(mapping_df)}")
    print(f"  Columns: {list(mapping_df.columns)}")
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    in_data    = mapping_df["in_training_data"].sum()
    is_0_5     = mapping_df["is_0_5_range"].sum()
    drops      = (mapping_df["type_3_strategy_code"] == "drop").sum()
    ohe        = (mapping_df["type_3_strategy_code"].str.startswith("ohe")).sum()
    ordered    = (mapping_df["type_3_strategy_code"] == "ordered").sum()
    binary     = (mapping_df["type_3_strategy_code"].str.startswith("binary")).sum()

    print("Summary:")
    print(f"  Features in training data  : {in_data}")
    print(f"  0-5 range features         : {is_0_5}")
    print(f"  Actuarial: Ordered group   : {ordered}")
    print(f"  Actuarial: Binary group    : {binary}")
    print(f"  Actuarial: One-Hot (ohe*)  : {ohe}")
    print(f"  Actuarial: Drop            : {drops}")
    print()

    # ── Save ──────────────────────────────────────────────────────────────────
    mapping_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Mapping reference saved to:\n   {OUTPUT_CSV}")
    print()
    print("Next steps:")
    print("  1. Review / edit config/level_mapping_reference.csv")
    print("  2. Fill in 'type_4_custom' and 'type_4_notes' columns if needed")
    print("  3. Run model_training.py to benchmark all encoding strategies")


if __name__ == "__main__":
    main()
