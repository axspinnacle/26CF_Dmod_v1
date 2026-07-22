"""
model_training.py
-----------------
Program 2 of 2 in the XGBoost benchmarking pipeline.

Handles model training, evaluation, and saving for each encoding experiment.

Usage (from main_xgboost_benchmark.ipynb):
    from model_training import train_and_evaluate, save_experiment, compare_results
"""

import os
import json
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── XGBoost import ────────────────────────────────────────────────────────────
try:
    import xgboost as xgb
except ImportError:
    raise ImportError("XGBoost not installed. Run: pip install xgboost")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
CONFIG_PATH  = os.path.join(PROJECT_ROOT, "config.json")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def _load_xgb_params() -> dict:
    """
    Load XGBoost hyperparameters from config.json → xgboost section.
    Falls back to safe defaults if section is missing.
    Keys starting with '_' (e.g. _note) are stripped automatically.
    base_score is NOT set here — computed dynamically from y_train.
    """
    _defaults = {
        "objective":        "reg:tweedie",
        "tweedie_variance_power": 1.5,
        "max_depth":        6,
        "learning_rate":    0.05,
        "n_estimators":     5000,
        "min_child_weight": 100,
        "subsample":        1.0,
        "colsample_bytree": 1.0,
        "alpha":            0,
        "tree_method":      "hist",
        "device":           "cpu",
        "random_state":     42,
        "verbosity":        1,
    }
    try:
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        params = cfg.get("xgboost", _defaults)
        # Strip comment keys (those starting with "_")
        return {k: v for k, v in params.items() if not k.startswith("_")}
    except Exception as e:
        print(f"  [Warning] Could not load xgboost config: {e}. Using defaults.")
        return _defaults


# Load once at module import — all 4 experiments share the same starting params
DEFAULT_XGB_PARAMS = _load_xgb_params()
print(f"  XGBoost config loaded: objective={DEFAULT_XGB_PARAMS.get('objective')}, "
      f"n_estimators={DEFAULT_XGB_PARAMS.get('n_estimators')}, "
      f"lr={DEFAULT_XGB_PARAMS.get('learning_rate')}")


def reload_config():
    """
    Re-read config.json and update DEFAULT_XGB_PARAMS in-place.
    Call this in the notebook before re-running an experiment after
    editing config.json, without needing to restart the kernel.

    Usage:
        from model_training import reload_config
        reload_config()
    """
    global DEFAULT_XGB_PARAMS
    DEFAULT_XGB_PARAMS = _load_xgb_params()
    print(f"  [reload] objective={DEFAULT_XGB_PARAMS.get('objective')}, "
          f"n_estimators={DEFAULT_XGB_PARAMS.get('n_estimators')}, "
          f"lr={DEFAULT_XGB_PARAMS.get('learning_rate')}, "
          f"tweedie_vp={DEFAULT_XGB_PARAMS.get('tweedie_variance_power')}")


# ============================================================================
# 1. Training and evaluation
# ============================================================================

def train_and_evaluate(
    experiment_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    feature_names: list,
    xgb_params: dict = None,
) -> dict:
    """
    Train an XGBoost model and evaluate on the test set.

    Parameters
    ----------
    experiment_name : str
        Short label, e.g. 'type1_ordinal', 'type2_binary'.
    X_train, y_train : training features and target.
    X_test, y_test   : test features and target.
    feature_names    : list of feature name strings (post-encoding).
    xgb_params       : dict of XGBoost parameters (defaults used if None).

    Returns
    -------
    dict with keys: experiment_name, model, feature_names, metrics,
                    y_pred, training_time_sec
    """
    params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}

    # base_score: set to mean of y_train (improves Tweedie convergence)
    # Only set if not already overridden by caller
    if "base_score" not in params:
        params["base_score"] = float(y_train.mean())

    print(f"\n  Training {experiment_name} ...")
    print(f"    X_train shape : {X_train.shape}")
    print(f"    X_test  shape : {X_test.shape}")
    print(f"    base_score    : {params['base_score']:.4f}  (mean of y_train)")

    # ── Fill NaN with -1 (XGBoost handles them, but explicit is cleaner) ─────
    X_train = X_train.fillna(-1)
    X_test  = X_test.fillna(-1)

    # ── Train ─────────────────────────────────────────────────────────────────
    t0    = time.time()
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    elapsed = round(time.time() - t0, 1)

    # ── Predict ───────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)

    # ── Metrics ───────────────────────────────────────────────────────────────
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))

    metrics = {
        "rmse":             round(rmse, 6),
        "mae":              round(mae,  6),
        "r2":               round(r2,   6),
        "training_time_sec": elapsed,
        "n_train":           len(y_train),
        "n_test":            len(y_test),
        "n_features":        len(feature_names),
    }

    print(f"    ✅ Done  |  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  "
          f"Time={elapsed}s")

    return {
        "experiment_name": experiment_name,
        "model":           model,
        "feature_names":   feature_names,
        "metrics":         metrics,
        "y_pred":          y_pred,
        "y_true":          y_test.values,
        "xgb_params":      params,
    }


# ============================================================================
# 2. Save experiment artifacts
# ============================================================================

def save_experiment(result: dict, encoders: dict = None) -> str:
    """
    Save model, feature names, predictions, metrics, and encoders
    for a single encoding experiment.

    Parameters
    ----------
    result   : dict returned by train_and_evaluate().
    encoders : dict of fitted encoders from encoding strategy (optional).

    Returns
    -------
    str : path to the experiment directory.
    """
    name    = result["experiment_name"]
    exp_dir = os.path.join(MODELS_DIR, f"pp_bi_{name}")
    os.makedirs(exp_dir, exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model_path = os.path.join(exp_dir, "xgboost_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(result["model"], f)

    # ── Feature names ─────────────────────────────────────────────────────────
    feat_path = os.path.join(exp_dir, "feature_names.json")
    with open(feat_path, "w") as f:
        json.dump(result["feature_names"], f, indent=2)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics_path = os.path.join(exp_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(result["metrics"], f, indent=2)

    # ── XGBoost params snapshot ───────────────────────────────────────────────
    params_path = os.path.join(exp_dir, "xgb_params.json")
    with open(params_path, "w") as f:
        json.dump(result["xgb_params"], f, indent=2)

    # ── Test predictions ──────────────────────────────────────────────────────
    pred_df = pd.DataFrame({
        "y_true": result["y_true"],
        "y_pred": result["y_pred"],
        "residual": result["y_true"] - result["y_pred"],
    })
    pred_df.to_csv(os.path.join(exp_dir, "test_predictions.csv"), index=False)

    # ── Encoders (optional) ───────────────────────────────────────────────────
    if encoders:
        enc_path = os.path.join(exp_dir, "encoders.pkl")
        with open(enc_path, "wb") as f:
            pickle.dump(encoders, f)

    print(f"    💾 Saved to: {exp_dir}/")
    return exp_dir


# ============================================================================
# 3. Compare all experiments
# ============================================================================

def compare_results(results: list) -> pd.DataFrame:
    """
    Build and display a summary comparison table for all experiments.

    Parameters
    ----------
    results : list of dicts returned by train_and_evaluate().

    Returns
    -------
    pd.DataFrame with one row per experiment.
    """
    rows = []
    for r in results:
        row = {"experiment": r["experiment_name"]}
        row.update(r["metrics"])
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("experiment")
    summary = summary.sort_values("rmse")

    # ── Identify winner ───────────────────────────────────────────────────────
    winner = summary["rmse"].idxmin()

    print("\n" + "=" * 65)
    print("  ENCODING BENCHMARK SUMMARY  (sorted by RMSE ↑ best)")
    print("=" * 65)
    print(summary.to_string())
    print(f"\n  🏆 Best strategy: {winner}  "
          f"(RMSE={summary.loc[winner,'rmse']:.4f})")
    print("=" * 65)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, "encoding_benchmark_summary.csv")
    summary.reset_index().to_csv(out_path, index=False)
    print(f"\n  📄 Summary saved: {out_path}")

    return summary


# ============================================================================
# 4. Load a saved model (for inference / test / holdout)
# ============================================================================

def load_experiment(experiment_name: str) -> dict:
    """
    Load a previously saved experiment from disk.

    Parameters
    ----------
    experiment_name : str, e.g. 'type1_ordinal'

    Returns
    -------
    dict with model, feature_names, metrics, encoders (if saved)
    """
    exp_dir = os.path.join(MODELS_DIR, f"pp_bi_{experiment_name}")
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"No saved experiment found at: {exp_dir}")

    with open(os.path.join(exp_dir, "xgboost_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(exp_dir, "feature_names.json")) as f:
        feature_names = json.load(f)
    with open(os.path.join(exp_dir, "metrics.json")) as f:
        metrics = json.load(f)

    encoders = None
    enc_path = os.path.join(exp_dir, "encoders.pkl")
    if os.path.exists(enc_path):
        with open(enc_path, "rb") as f:
            encoders = pickle.load(f)

    return {
        "experiment_name": experiment_name,
        "model":           model,
        "feature_names":   feature_names,
        "metrics":         metrics,
        "encoders":        encoders,
    }
