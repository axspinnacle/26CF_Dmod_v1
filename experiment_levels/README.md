# Experiment Levels - XGBoost Benchmark Project

## Memory Optimization Status ✅

### **CONFIRMED: Dtype Optimization IS Implemented**

All data loading functions in `code/encoding_strategies.py` are using dtype optimization to reduce memory usage by ~50%.

#### Implementation Details

**1. Optimization Function** (`code/data_processing.py`, lines 299-326)
```python
def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory usage.
    
    Conversions:
    - float64 → float32 (50% memory reduction)
    - int64 → int32 (50% memory reduction)
    """
```

**2. Loading Functions Using Optimization** (`code/encoding_strategies.py`)

| Function | Line | Status |
|----------|------|--------|
| `load_train_only()` | 111 | ✅ Optimized |
| `load_test_only()` | 142 | ✅ Optimized |
| `load_train_test()` | 167, 175 | ✅ Optimized (both train & test) |

**3. Lazy Import Pattern** (lines 75-80)
```python
def _optimize():
    """Lazy import of optimize_dtypes to avoid circular dependency."""
    from data_processing import optimize_dtypes
    return optimize_dtypes
```

#### Memory Savings

For typical datasets with numeric columns:
- **Before optimization**: 8 GB → **After**: ~4 GB
- **Reduction**: 40-50% memory usage

#### Loading Strategy

**For memory-constrained environments (recommended for full data):**
```python
# Train-only workflow (separate notebooks)
train = load_train_only(debug=0)  # Full data, optimized
# Train models...
# Then in separate notebook:
test = load_test_only(debug=0)    # Evaluate models
```

**For sufficient memory (benchmark/development):**
```python
# Load both simultaneously
train, test = load_train_test(debug=0)
```

#### Debug Modes

All functions support debug parameter for memory-efficient testing:
- `debug=0`: Full dataset
- `debug=1`: 10K train / 2K test rows (fast smoke test)
- `debug=2`: 10% random sample (medium-scale validation)

---

## Project Structure

### Notebooks
- `main_train_only.ipynb` - Training workflow (memory-optimized)
- `main_test_eval.ipynb` - Evaluation workflow (separate from training)
- `main_xgboost_benchmark_full.ipynb` - Full benchmark (requires more memory)
- `main_splits.ipynb` - Data splitting experiments
- `main.ipynb` - Legacy main notebook

### Code Modules
- `code/data_processing.py` - Data loading, optimization, preprocessing
- `code/encoding_strategies.py` - Categorical encoding (ordinal, binary, actuarial)
- `code/model_training.py` - XGBoost training and evaluation
- `code/visualization.py` - Lift charts, SHAP plots
- `code/create_level_mapping.py` - Level mapping utilities

### Results
- `models/pp_bi_type{1,2,3}_{encoding}/` - Trained models by encoding type
- `results/encoding_benchmark_summary.csv` - Performance comparison
- `results/actuarial_metrics_summary.csv` - Actuarial metrics
- `results/lift_charts/` - Model lift visualizations

### Configuration
- `config.json` - Project paths and settings
- `config/level_mapping_reference.csv` - Reference mappings

---

## Encoding Strategies Benchmarked

1. **Type 1 - Ordinal Encoding**
   - Keep original 0-5 integer levels unchanged
   - Non-0-5 features pass through as-is
   - Fastest, lowest memory

2. **Type 2 - Binary Encoding**
   - Collapse 0-5 features using {0,1,2}->0 | {3,4,5}->1
   - String features: One-Hot Encoded
   - Compact binary representation

3. **Type 3 - Actuarial Encoding**
   - Per-column strategy from actuary's specification
   - Strategies: ordered, binary_low_hi, binary_lo_high, ohe, h_map2to4, etc.
   - Best predictive performance

4. **Type 4 - Custom Encoding (NEW)**
   - **Step 1**: Remap level 2 → level 4 for all 0-5 features
   - **Step 2**: Apply binary grouping {0,1,4}->0 | {3,5}->1
   - **Result**: Features have only levels {0, 1, 3, 4, 5} — no level 2
   - String features: One-Hot Encoded
   - Use case: Test hierarchical remapping + custom binary grouping

---

*Last updated: 2026-07-22*
