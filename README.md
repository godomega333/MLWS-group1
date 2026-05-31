# Machine Learning Workshop Group 1 Code Package

This package contains the code, data, and selected experiment outputs for the
Group 1 Spaceship Titanic project. The task is a Kaggle binary classification
problem: predict whether each passenger was transported.

The best submitted model in this package is the final CatBoost model in
`models/cat-boost/catboost_final.py`, using threshold `0.4725`. The recorded
public leaderboard score for this submission is `0.81973`.

## Package Structure

```text
workshop-group1/
├── environment.yml
├── spaceship-titanic-dataset/
│   ├── train.csv
│   ├── test.csv
│   └── sample-submission.csv
├── models/
│   ├── logistic-regression/
│   ├── random-forest/
│   ├── xg-boost/
│   └── cat-boost/
└── experiments/
    ├── scripts/
    └── results/
```

- `spaceship-titanic-dataset/` contains the raw Kaggle data.
- `models/` contains the four model families used in the project.
- `experiments/scripts/` contains validation, ablation, ensemble, and threshold
  analysis scripts.
- `experiments/results/` contains generated result tables and submission files
  used by the analysis.

All scripts use relative paths. Run commands from the package root directory
`workshop-group1`.

## Environment Setup

The recommended environment manager is Conda.

### macOS / Linux

```bash
cd /path/to/workshop-group1
conda env create -f environment.yml
conda activate ml-project
```

If the environment already exists:

```bash
cd /path/to/workshop-group1
conda env update -f environment.yml --prune
conda activate ml-project
```

### Windows PowerShell

```powershell
cd C:\path\to\workshop-group1
conda env create -f environment.yml
conda activate ml-project
```

If the environment already exists:

```powershell
cd C:\path\to\workshop-group1
conda env update -f environment.yml --prune
conda activate ml-project
```

## Quick Checks

After activating the environment, run:

### macOS / Linux

```bash
python -c "from pathlib import Path; print(Path('spaceship-titanic-dataset/train.csv').exists())"
python experiments/scripts/rank_threshold.py
```

### Windows PowerShell

```powershell
python -c "from pathlib import Path; print(Path('spaceship-titanic-dataset/train.csv').exists())"
python experiments/scripts/rank_threshold.py
```

The first command should print `True`. The second command prints the rank of
the selected CatBoost threshold within the stored OOF threshold sweep.

## Main Reproduction Commands

### Final CatBoost Submission

This is the strongest submitted model in the package.

macOS / Linux:

```bash
python models/cat-boost/catboost_final.py
```

Windows PowerShell:

```powershell
python models/cat-boost/catboost_final.py
```

Outputs:

- `models/cat-boost/outputs/submission_catboost_final.csv`
- `models/cat-boost/outputs/test_proba_catboost_final.csv`

### Four Model-Family Comparison Runs

macOS / Linux:

```bash
python models/logistic-regression/lr_comparison.py
python models/random-forest/rf_comparison.py
python models/xg-boost/xgboost_comparison.py
python models/cat-boost/catboost_final.py
```

Windows PowerShell:

```powershell
python models/logistic-regression/lr_comparison.py
python models/random-forest/rf_comparison.py
python models/xg-boost/xgboost_comparison.py
python models/cat-boost/catboost_final.py
```

Each script writes its outputs under the corresponding `models/<model>/outputs/`
folder.

### Final Experiment Tables

macOS / Linux:

```bash
python experiments/scripts/run_experiments.py
```

Windows PowerShell:

```powershell
python experiments/scripts/run_experiments.py
```

This script writes validation, ablation, calibration, SHAP, and McNemar outputs
under `experiments/results/`. It can take several minutes because it retrains
multiple models.

To run only selected experiment groups:

```bash
python experiments/scripts/run_experiments.py --tasks group_cv eps_ablation feature_ablation
```

The same command works in Windows PowerShell.

### Ensemble Experiment

macOS / Linux:

```bash
python experiments/scripts/run_ensemble.py
```

Windows PowerShell:

```powershell
python experiments/scripts/run_ensemble.py
```

This script uses the group-aware OOF probability files in `experiments/results/`
and writes ensemble outputs to the same folder.

### Threshold Analysis

macOS / Linux:

```bash
python experiments/scripts/rank_threshold.py
python experiments/scripts/plot_threshold_landscape.py
```

Windows PowerShell:

```powershell
python experiments/scripts/rank_threshold.py
python experiments/scripts/plot_threshold_landscape.py
```

The plot script writes `experiments/figures/threshold_landscape.png`.

## Supplementary Experiments

The following scripts reproduce additional perturbation experiments:

```bash
python experiments/scripts/run_raw_cabin_baseline.py
python experiments/scripts/run_e2_a_cryosleep_spend_rule.py
python experiments/scripts/run_e2_c_family_zero_flags.py
python experiments/scripts/run_e2_g_lightgbm.py
```

The same commands work in Windows PowerShell. These scripts write predictions,
probabilities, and summary files under `experiments/results/`.

## Optional Search Scripts

The randomized search scripts are included for completeness:

```bash
python models/logistic-regression/lr_grid.py
python models/random-forest/rf_grid.py
python models/xg-boost/xgboost_grid.py
```

The CatBoost search/probe scripts use Modal:

```bash
modal run models/cat-boost/catboost_grid_search.py
modal run models/cat-boost/catboost_iteration_probe.py
modal run models/cat-boost/catboost_l2_depth_ablation.py
```

The Modal commands require a configured Modal account and are not required to
run the final CatBoost submission.

## Notes

- Run scripts from the `workshop-group1` root directory.
- Do not edit the raw CSV files in `spaceship-titanic-dataset/`.
- Kaggle public leaderboard scores require submitting generated CSV files to
  the Kaggle Spaceship Titanic competition page.
- Some full experiment scripts may take several minutes depending on hardware.
