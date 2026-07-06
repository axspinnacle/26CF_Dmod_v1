# Depreciation Model Data Pipeline

Consolidated pipeline combining offset calculation, superpolicy consolidation,
fold assignment, and final data combination. Uses ONE `config.json` with
separate sections per step so config can be reused/copied to other codebases.

## Structure

```
main_folder/
├── config.json                 # single config, sectioned: global, calc_offset, add_superpolicy, add_fold, combine_data
├── shared/
│   └── config_loader.py
├── 01_calc_offset/             # imputation + GLM dep factor prediction
├── 02_add_superpolicy/         # consolidate policies sharing a VIN (independent of 01)
├── 03_add_fold/                # assign 10-fold CV splits (depends on 02's output)
├── 04_combine_data/            # join dep factor (01) + fold (03) output
└── tests/                      # compare new vs old pipeline outputs
```

## Execution order

```
                 ┌── 01 Calc Offset ────────────┐
raw master data ─┤                              ├─► 04 Combine ─► final output
                 └── 02 Add Superpolicy ─► 03 Add Fold ─┘
```

- 01 and 02 can run in parallel (both only need raw master data).
- 03 requires 02's output (superpolicy_id).
- 04 requires both 01's output (dep factor) and 03's output (fold).

## Setup on a new machine

1. Copy the `main_folder` directory to the new machine.
2. Edit `config.json` -> `global.base_data_dir` to point to the data folder on
   the new machine. All other paths in the config are relative to this.
3. Run notebooks in order: `01_calc_offset/run_offset.ipynb`,
   `02_add_superpolicy/run_superpolicy.ipynb`,
   `03_add_fold/run_folds.ipynb`, `04_combine_data/run_combine.ipynb`.
4. Use the `_debug` notebook variants if you want diagnostics/plots at each
   step (missing value checks, histograms, fold validation, etc.)

## Reusing a single step elsewhere

Since the config is sectioned, you can copy just the `global` + one step's
section (e.g. `calc_offset`) into a new config file for use in another
codebase, along with that step's folder.

## Testing

`tests/` holds notebooks that compare the new pipeline's output against the
original scattered notebooks' output, using `compare_utils.py` (row/column
diffs, value count comparisons, numeric diff summaries).
