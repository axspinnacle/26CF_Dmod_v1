"""
visualization.py
----------------
Lift chart and model comparison plots for the XGBoost benchmarking pipeline.

Adapted from: code/reference_folder/A7_read100_parq.ipynb
  - lift_chart_modified_v2  -> build_lift_data / lift_chart
  - model_metrics_modified  -> model_metrics
  - compare_lift_charts     -> NEW: overlays all 4 experiments

Usage (from main_xgboost_benchmark.ipynb):
    from visualization import lift_chart, model_metrics, compare_lift_charts
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH  = os.path.join(PROJECT_ROOT, "config.json")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
os.makedirs(os.path.join(RESULTS_DIR, "lift_charts"), exist_ok=True)

with open(CONFIG_PATH) as f:
    _cfg = json.load(f)

LIFT_BINS    = _cfg["modeling"].get("lift_bins", 10)
EXPOSURE_COL = _cfg["modeling"].get("exposure_column", None)  # e.g. "ee_bi"


# ============================================================================
# 1. Core helpers
# ============================================================================

def _build_lift_data(y_true: np.ndarray, y_pred: np.ndarray,
                     weights: np.ndarray, bins: int) -> pd.DataFrame:
    """
    Build a decile table sorted by predicted value (ascending).
    Weights are used for exposure-weighted actual and predicted PP.

    Columns returned: decile, weight, act, pred
    """
    df = pd.DataFrame({
        "y_true":  y_true,
        "y_pred":  y_pred,
        "weight":  weights,
    })
    df["act_weighted"]  = df["weight"] * df["y_true"]
    df["pred_weighted"] = df["weight"] * df["y_pred"]

    # Weight-based cumulative decile assignment (ascending predicted)
    df = df.sort_values("y_pred", ascending=True)
    cumw = df["weight"].cumsum() / df["weight"].sum()
    df["decile"] = (round(cumw, 2) * bins).apply(np.floor)
    df["decile"] = np.where(df["decile"] + 1 > bins, bins, df["decile"] + 1)

    x = (df.groupby("decile", dropna=False)
           .agg(weight=("weight", "sum"),
                act_weighted=("act_weighted", "sum"),
                pred_weighted=("pred_weighted", "sum"))
           .reset_index())

    x["act"]  = x["act_weighted"]  / x["weight"]
    x["pred"] = x["pred_weighted"] / x["weight"]
    x.drop(columns=["act_weighted", "pred_weighted"], inplace=True)
    return x


# ============================================================================
# 2. Lift chart — single model
# ============================================================================

def lift_chart(result: dict, test_df: pd.DataFrame = None,
               bins: int = None, print_table: bool = True,
               title: str = None, save: bool = True) -> pd.DataFrame:
    """
    Draw a lift chart for one experiment result.

    Parameters
    ----------
    result   : dict returned by model_training.train_and_evaluate()
    test_df  : raw test DataFrame (used to extract exposure weights if available)
    bins     : number of decile bins (default from config.json -> modeling.lift_bins)
    print_table : whether to print the decile table
    title    : chart title (auto-generated if None)
    save     : if True, save PNG to results/lift_charts/

    Returns
    -------
    pd.DataFrame : decile table (decile, weight, act, pred)
    """
    bins   = bins or LIFT_BINS
    name   = result["experiment_name"]
    y_true = result["y_true"].astype(float)
    y_pred = result["y_pred"].astype(float)

    # ── Weights: use exposure column if available, else equal weights ─────────
    if test_df is not None and EXPOSURE_COL and EXPOSURE_COL in test_df.columns:
        weights = test_df[EXPOSURE_COL].values.astype(float)
        wt_label = EXPOSURE_COL
    else:
        weights = np.ones(len(y_true))
        wt_label = "equal_weight"

    lift_df = _build_lift_data(y_true, y_pred, weights, bins)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    y_max = max(lift_df["act"].max(), lift_df["pred"].max()) * 1.20
    ax2.set_ylim(0, y_max)

    lift_df["weight"].plot.bar(ax=ax1, alpha=0.4, color="steelblue", label="Exposure")
    lift_df["act"].plot( ax=ax2, marker="o", linewidth=0,   color="black",  label="Actual")
    lift_df["pred"].plot(ax=ax2, marker="x", linestyle="--", color="red",  label="Predicted")

    ax1.set_xlabel("Decile (sorted by predicted, ascending)")
    ax1.set_ylabel("Exposure weight")
    ax2.set_ylabel("Pure Premium")
    ax2.legend(loc="upper left")
    ax1.legend(loc="upper right")
    chart_title = title or f"Lift Chart – {name}  (weight: {wt_label}, bins={bins})"
    plt.title(chart_title)
    plt.tight_layout()

    if save:
        out = os.path.join(RESULTS_DIR, "lift_charts", f"lift_{name}.png")
        plt.savefig(out, dpi=120)
        print(f"  💾 Lift chart saved: {out}")

    plt.show()

    if print_table:
        print(lift_df.to_string(index=False))

    return lift_df


# ============================================================================
# 3. Model metrics (actuarial: model power + fit quality)
# ============================================================================

def model_metrics(result: dict, test_df: pd.DataFrame = None,
                  bins: int = 50) -> dict:
    """
    Compute actuarial model metrics.

    Model Power  = weighted mean absolute deviation of decile PP from overall mean PP.
    Fit Quality  = 1 – weighted mean of |pred/act – 1| per decile.

    Returns dict: {model_power, fit_quality, experiment_name}
    """
    y_true = result["y_true"].astype(float)
    y_pred = result["y_pred"].astype(float)

    if test_df is not None and EXPOSURE_COL and EXPOSURE_COL in test_df.columns:
        weights = test_df[EXPOSURE_COL].values.astype(float)
    else:
        weights = np.ones(len(y_true))

    x = _build_lift_data(y_true, y_pred, weights, bins)

    tot_pp = x["act_weighted"].sum() / x["weight"].sum() if "act_weighted" in x.columns else (
        (x["act"] * x["weight"]).sum() / x["weight"].sum()
    )

    # Fit quality: 1 – Σ(|pred/act – 1| × w) / Σw
    x["decile_err"]    = (abs(x["pred"] / x["act"] - 1)
                          .replace([np.inf, -np.inf], np.nan)
                          .fillna(1))
    x["decile_err_sp"] = x["decile_err"] * x["weight"]
    fit_quality = max(0.0, 1.0 - x["decile_err_sp"].sum() / x["weight"].sum())

    # Model power: Σ(|pred/mean – 1| × w) / Σw
    x["diff_unity"] = abs(x["pred"] / tot_pp - 1) * x["weight"]
    model_power = x["diff_unity"].sum() / x["weight"].sum()

    return {
        "experiment_name": result["experiment_name"],
        "model_power":     round(model_power, 6),
        "fit_quality":     round(fit_quality, 6),
    }


# ============================================================================
# 4. Compare all experiments — overlay lift curves
# ============================================================================

def compare_lift_charts(results: list, test_df: pd.DataFrame = None,
                        bins: int = None, save: bool = True) -> pd.DataFrame:
    """
    Overlay lift curves for all encoding experiments on one chart.

    Parameters
    ----------
    results  : list of dicts from model_training.train_and_evaluate()
    test_df  : raw test DataFrame for exposure weights
    bins     : number of decile bins
    save     : save PNG to results/lift_charts/comparison_lift.png

    Returns
    -------
    pd.DataFrame : metrics summary (experiment, model_power, fit_quality, rmse)
    """
    bins = bins or LIFT_BINS
    y_max_global = 0.0
    lift_tables = {}
    metrics_rows = []

    if test_df is not None and EXPOSURE_COL and EXPOSURE_COL in test_df.columns:
        weights = test_df[EXPOSURE_COL].values.astype(float)
    else:
        weights = np.ones(len(results[0]["y_true"]))

    # ── Pre-compute all lift tables ───────────────────────────────────────────
    for r in results:
        lt = _build_lift_data(r["y_true"].astype(float),
                              r["y_pred"].astype(float),
                              weights, bins)
        lift_tables[r["experiment_name"]] = lt
        y_max_global = max(y_max_global, lt["act"].max(), lt["pred"].max())
        mmet = model_metrics(r, test_df, bins=bins)
        mmet["rmse"] = r["metrics"]["rmse"]
        metrics_rows.append(mmet)

    # ── Plot ──────────────────────────────────────────────────────────────────
    colors = cm.tab10.colors
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_ylim(0, y_max_global * 1.20)

    # Plot actual (should be same for all) once
    first = list(lift_tables.values())[0]
    ax.plot(first["decile"], first["act"], marker="o", linewidth=0,
            color="black", markersize=8, label="Actual", zorder=5)

    for i, (name, lt) in enumerate(lift_tables.items()):
        ax.plot(lt["decile"], lt["pred"], marker="x", linestyle="--",
                color=colors[i % len(colors)], label=f"Pred – {name}")

    ax.set_xlabel("Decile (sorted by predicted, ascending)")
    ax.set_ylabel("Pure Premium")
    ax.set_title(f"Lift Chart Comparison — All Encoding Strategies  (bins={bins})")
    ax.legend(loc="upper left")
    plt.tight_layout()

    if save:
        out = os.path.join(RESULTS_DIR, "lift_charts", "comparison_lift.png")
        plt.savefig(out, dpi=120)
        print(f"  💾 Comparison lift chart saved: {out}")

    plt.show()

    # ── Metrics summary ───────────────────────────────────────────────────────
    summary = pd.DataFrame(metrics_rows).set_index("experiment_name")
    summary = summary.sort_values("model_power", ascending=False)

    print("\n" + "=" * 65)
    print("  ACTUARIAL METRICS COMPARISON")
    print("=" * 65)
    print(summary.to_string())
    print("=" * 65)

    out_csv = os.path.join(RESULTS_DIR, "actuarial_metrics_summary.csv")
    summary.reset_index().to_csv(out_csv, index=False)
    print(f"\n  📄 Actuarial metrics saved: {out_csv}")

    return summary
