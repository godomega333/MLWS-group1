"""S1 experiments runner.

Runs every locked S1 experiment and writes artifacts under experiments/results/.
Each experiment is a separate function and can be invoked individually via CLI.

  python experiments/scripts/run_experiments.py --tasks group_cv eps_ablation \
      feature_ablation cost shap mcnemar calibration

Default --tasks runs the full sequence in the order above.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time

import numpy as np
import pandas as pd
import psutil
from catboost import CatBoostClassifier
from scipy import stats as scipy_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from _pipeline import (
    CB_PARAMS, CB_THRESHOLD, DEFAULT_DATA_DIR, FIGURES_DIR, LR_PARAMS, RESULTS_DIR,
    RF_PARAMS, SEED, TABLES_DIR, XGB_PARAMS, build_lr_frame, build_tree_frame,
)


N_SPLITS = 5
SPLIT_RANDOM_STATE = 42
ALL_MODELS = ("logistic_regression", "random_forest", "xgboost", "catboost")
# Active data dir; mutated by main() if --data-dir is passed. Each experiment
# function reads this rather than relying on `build_tree_frame`'s default,
# because Python binds default args at function-definition time.
DATA_DIR = DEFAULT_DATA_DIR
DROP_OPTIONS = (
    ("none", []),
    ("drop_topology_cluster", ["Topology_Cluster"]),
    ("drop_cryosleep", ["CryoSleep"]),
    ("drop_spending", ["__SPENDING__"]),
    ("drop_side", ["Side"]),
    ("drop_deck", ["Deck"]),
)
EPS_GRID = (0.45, 0.55, 0.60, 0.666, 0.75, 0.85, 1.00)


# ---------------------------------------------------------------------------
# Helpers


def ensure_dirs():
    for d in (RESULTS_DIR, FIGURES_DIR, TABLES_DIR):
        os.makedirs(d, exist_ok=True)


def stratified_group_kfold():
    return StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SPLIT_RANDOM_STATE)


def stratified_kfold():
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SPLIT_RANDOM_STATE)


def _peak_rss_during(callable_fn, sample_interval_s=0.05):
    """Run callable_fn() while a background thread samples process RSS.

    Returns (result, peak_rss_delta_kb) where peak_rss_delta_kb =
        max(RSS during call) - RSS measured immediately before the call,
    in KB. Captures native allocations from C++ libraries (CatBoost / XGBoost
    / sklearn RF) that tracemalloc would miss.
    """
    proc = psutil.Process()
    baseline = proc.memory_info().rss
    peak = baseline
    stop = threading.Event()

    def sample():
        nonlocal peak
        while not stop.is_set():
            try:
                rss = proc.memory_info().rss
                if rss > peak:
                    peak = rss
            except psutil.Error:
                pass
            stop.wait(sample_interval_s)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        result = callable_fn()
    finally:
        stop.set()
        sampler.join(timeout=1.0)
    return result, max(peak - baseline, 0) / 1024.0


def _ci_95_mean(values):
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return float(arr.mean()), 0.0
    mean = float(arr.mean())
    sem = float(arr.std(ddof=1) / np.sqrt(arr.size))
    half = 1.96 * sem
    return mean, half


# ---------------------------------------------------------------------------
# Per-model training closures


def _fit_lr(X_tr, y_tr, params):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    model = LogisticRegression(**params)
    model.fit(X_tr_s, y_tr)
    return model, scaler


def _predict_lr(model_scaler, X):
    model, scaler = model_scaler
    return model.predict_proba(scaler.transform(X))[:, 1]


def _fit_rf(X_tr, y_tr, params):
    model = RandomForestClassifier(**params)
    model.fit(X_tr, y_tr)
    return model


def _predict_rf(model, X):
    return model.predict_proba(X)[:, 1]


def _fit_xgb(X_tr, y_tr, params):
    model = XGBClassifier(**params)
    model.fit(X_tr, y_tr)
    return model


def _predict_xgb(model, X):
    return model.predict_proba(X)[:, 1]


def _fit_cb(X_tr, y_tr, params, cat_features):
    model = CatBoostClassifier(**params, cat_features=cat_features)
    model.fit(X_tr, y_tr)
    return model


def _predict_cb(model, X):
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Experiment 1 + 4: Group-aware CV plus computational cost


def run_group_aware_cv():
    """Group-aware 5-fold CV for all four models, plus stratified CV for gap.

    Saves OOF predictions and per-fold metrics + cost; group-aware OOF probabilities
    are reused later for McNemar and calibration.
    """
    print("[exp1+4] building feature frames")
    tree = build_tree_frame(data_dir=DATA_DIR)
    lr = build_lr_frame(data_dir=DATA_DIR)

    n_train = len(tree.train)
    y = tree.y_train
    groups = tree.train_groups

    sgkf = stratified_group_kfold()
    skf = stratified_kfold()

    # --- Group-aware CV ---
    fold_records = []
    cost_records = []
    oof_proba = {m: np.zeros(n_train, dtype=float) for m in ALL_MODELS}
    oof_pred = {m: np.zeros(n_train, dtype=int) for m in ALL_MODELS}
    oof_fold_id = np.full(n_train, -1, dtype=int)

    for fold_idx, (tr_idx, va_idx) in enumerate(sgkf.split(tree.train, y, groups)):
        oof_fold_id[va_idx] = fold_idx
        print(f"[exp1] fold {fold_idx + 1}/{N_SPLITS} train={len(tr_idx)} val={len(va_idx)}")

        for model_name in ALL_MODELS:
            if model_name == "logistic_regression":
                X_tr = lr.train[lr.features].iloc[tr_idx].astype(float).values
                X_va = lr.train[lr.features].iloc[va_idx].astype(float).values
                t0 = time.perf_counter()
                (model_scaler), peak_rss_kb = _peak_rss_during(lambda: _fit_lr(X_tr, y[tr_idx], LR_PARAMS))
                fit_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                proba = _predict_lr(model_scaler, X_va)
                infer_s = time.perf_counter() - t0
            else:
                X_tr = tree.train[tree.features].iloc[tr_idx]
                X_va = tree.train[tree.features].iloc[va_idx]
                if model_name == "random_forest":
                    t0 = time.perf_counter()
                    (model), peak_rss_kb = _peak_rss_during(lambda: _fit_rf(X_tr, y[tr_idx], RF_PARAMS))
                    fit_s = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    proba = _predict_rf(model, X_va)
                    infer_s = time.perf_counter() - t0
                elif model_name == "xgboost":
                    t0 = time.perf_counter()
                    (model), peak_rss_kb = _peak_rss_during(lambda: _fit_xgb(X_tr, y[tr_idx], XGB_PARAMS))
                    fit_s = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    proba = _predict_xgb(model, X_va)
                    infer_s = time.perf_counter() - t0
                else:  # catboost
                    t0 = time.perf_counter()
                    (model), peak_rss_kb = _peak_rss_during(lambda: _fit_cb(X_tr, y[tr_idx], CB_PARAMS, tree.cat_features))
                    fit_s = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    proba = _predict_cb(model, X_va)
                    infer_s = time.perf_counter() - t0

            pred = (proba >= 0.5).astype(int)
            if model_name == "catboost":
                pred = (proba >= CB_THRESHOLD).astype(int)
            acc = accuracy_score(y[va_idx], pred)
            oof_proba[model_name][va_idx] = proba
            oof_pred[model_name][va_idx] = pred

            fold_records.append({
                "model": model_name,
                "fold": fold_idx,
                "n_val": len(va_idx),
                "acc": acc,
                "fit_seconds": fit_s,
                "infer_seconds": infer_s,
                "peak_rss_kb": peak_rss_kb,
            })
            print(f"        [exp1] {model_name:<19s} acc={acc:.5f}  fit={fit_s:.2f}s  pred={infer_s*1000:.1f}ms  peak_rss_delta={peak_rss_kb/1024:.1f}MiB")

        cost_records.append({"fold": fold_idx, "n_val": len(va_idx)})

    fold_df = pd.DataFrame(fold_records)
    fold_df.to_csv(os.path.join(RESULTS_DIR, "cv_group_aware_per_fold.csv"), index=False)

    summary_rows = []
    for model_name in ALL_MODELS:
        m = fold_df[fold_df["model"] == model_name]
        mean_acc, ci = _ci_95_mean(m["acc"].values)
        summary_rows.append({
            "model": model_name,
            "group_aware_mean_acc": mean_acc,
            "group_aware_ci95_half": ci,
            "group_aware_oof_acc": float(accuracy_score(y, oof_pred[model_name])),
            "fit_total_seconds": float(m["fit_seconds"].sum()),
            "infer_total_seconds": float(m["infer_seconds"].sum()),
            "peak_rss_delta_mib_max": float(m["peak_rss_kb"].max() / 1024),
        })
    summary_df = pd.DataFrame(summary_rows)

    # OOF storage
    for model_name in ALL_MODELS:
        pd.DataFrame({
            "row_index": np.arange(n_train),
            "fold": oof_fold_id,
            "y_true": y,
            "proba": oof_proba[model_name],
            "pred": oof_pred[model_name],
        }).to_csv(os.path.join(RESULTS_DIR, f"oof_group_aware_{model_name}.csv"), index=False)

    # --- Stratified CV for gap comparison (uses same final-config hyperparameters) ---
    print("[exp1] running stratified KFold for ID-vs-OOD gap")
    strat_records = []
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(tree.train, y)):
        for model_name in ALL_MODELS:
            if model_name == "logistic_regression":
                X_tr = lr.train[lr.features].iloc[tr_idx].astype(float).values
                X_va = lr.train[lr.features].iloc[va_idx].astype(float).values
                model_scaler = _fit_lr(X_tr, y[tr_idx], LR_PARAMS)
                proba = _predict_lr(model_scaler, X_va)
            else:
                X_tr = tree.train[tree.features].iloc[tr_idx]
                X_va = tree.train[tree.features].iloc[va_idx]
                if model_name == "random_forest":
                    model = _fit_rf(X_tr, y[tr_idx], RF_PARAMS)
                    proba = _predict_rf(model, X_va)
                elif model_name == "xgboost":
                    model = _fit_xgb(X_tr, y[tr_idx], XGB_PARAMS)
                    proba = _predict_xgb(model, X_va)
                else:
                    model = _fit_cb(X_tr, y[tr_idx], CB_PARAMS, tree.cat_features)
                    proba = _predict_cb(model, X_va)
            thr = CB_THRESHOLD if model_name == "catboost" else 0.5
            pred = (proba >= thr).astype(int)
            strat_records.append({
                "model": model_name, "fold": fold_idx, "acc": accuracy_score(y[va_idx], pred),
            })
    strat_df = pd.DataFrame(strat_records)
    strat_df.to_csv(os.path.join(RESULTS_DIR, "cv_stratified_per_fold.csv"), index=False)

    strat_summary = strat_df.groupby("model")["acc"].agg(["mean"]).rename(columns={"mean": "stratified_mean_acc"})
    summary_df = summary_df.merge(strat_summary, left_on="model", right_index=True)
    summary_df["id_minus_ood_gap"] = summary_df["stratified_mean_acc"] - summary_df["group_aware_mean_acc"]
    summary_df.to_csv(os.path.join(RESULTS_DIR, "cv_summary.csv"), index=False)

    print(summary_df.round(5).to_string(index=False))
    return summary_df


# ---------------------------------------------------------------------------
# Experiment 2: DBSCAN eps ablation (CatBoost group-aware CV)


def run_eps_ablation():
    print("[exp2] DBSCAN eps ablation (CatBoost, group-aware 5-fold CV)")
    rows = []
    for eps in EPS_GRID:
        tree = build_tree_frame(data_dir=DATA_DIR, eps=eps, min_samples=2)
        y = tree.y_train
        sgkf = stratified_group_kfold()
        fold_accs = []
        for fold_idx, (tr_idx, va_idx) in enumerate(sgkf.split(tree.train, y, tree.train_groups)):
            X_tr = tree.train[tree.features].iloc[tr_idx]
            X_va = tree.train[tree.features].iloc[va_idx]
            model = _fit_cb(X_tr, y[tr_idx], CB_PARAMS, tree.cat_features)
            proba = _predict_cb(model, X_va)
            pred = (proba >= CB_THRESHOLD).astype(int)
            fold_accs.append(accuracy_score(y[va_idx], pred))
        mean, ci = _ci_95_mean(fold_accs)
        n_clusters = int(pd.Series(tree.train["Topology_Cluster"]).nunique())
        n_noise = int((tree.train["Topology_Cluster"] == -1).sum())
        rows.append({
            "eps": eps,
            "min_samples": 2,
            "n_clusters_train": n_clusters,
            "n_noise_train": n_noise,
            "group_aware_mean_acc": mean,
            "group_aware_ci95_half": ci,
            "fold_accs": json.dumps([float(x) for x in fold_accs]),
        })
        print(f"        [exp2] eps={eps:.3f}  acc={mean:.5f} +/- {ci:.5f}  clusters={n_clusters}  noise={n_noise}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "eps_ablation_catboost.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# Experiment 3: Feature ablation (CatBoost group-aware CV)


def run_feature_ablation():
    print("[exp3] feature ablation (CatBoost, group-aware 5-fold CV)")
    rows = []
    baseline_acc = None
    for label, drop in DROP_OPTIONS:
        tree = build_tree_frame(data_dir=DATA_DIR, eps=0.666, min_samples=2, drop_features=drop)
        y = tree.y_train
        sgkf = stratified_group_kfold()
        fold_accs = []
        for fold_idx, (tr_idx, va_idx) in enumerate(sgkf.split(tree.train, y, tree.train_groups)):
            X_tr = tree.train[tree.features].iloc[tr_idx]
            X_va = tree.train[tree.features].iloc[va_idx]
            model = _fit_cb(X_tr, y[tr_idx], CB_PARAMS, tree.cat_features)
            proba = _predict_cb(model, X_va)
            pred = (proba >= CB_THRESHOLD).astype(int)
            fold_accs.append(accuracy_score(y[va_idx], pred))
        mean, ci = _ci_95_mean(fold_accs)
        if label == "none":
            baseline_acc = mean
        delta = mean - baseline_acc if baseline_acc is not None else 0.0
        rows.append({
            "drop_label": label,
            "drop_features": json.dumps(list(drop)),
            "n_features": len(tree.features),
            "group_aware_mean_acc": mean,
            "group_aware_ci95_half": ci,
            "delta_vs_baseline": delta,
            "fold_accs": json.dumps([float(x) for x in fold_accs]),
        })
        print(f"        [exp3] {label:<24s} acc={mean:.5f} +/- {ci:.5f}  delta={delta:+.5f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "feature_ablation_catboost.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# Experiment 5: SHAP on final CatBoost (full-fit)


def run_shap_analysis():
    print("[exp5] SHAP analysis on final CatBoost (full-fit on train)")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    from matplotlib import font_manager

    tree = build_tree_frame(data_dir=DATA_DIR)
    y = tree.y_train

    model = _fit_cb(tree.train[tree.features], y, CB_PARAMS, tree.cat_features)

    # SHAP values via CatBoost's tree-aware explainer.
    sample_size = min(2000, len(tree.train))
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(tree.train), size=sample_size, replace=False)
    X_sample = tree.train[tree.features].iloc[sample_idx]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    abs_mean = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": tree.features,
        "mean_abs_shap": abs_mean,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance_df.to_csv(os.path.join(RESULTS_DIR, "shap_global_importance.csv"), index=False)

    # Save raw SHAP values + feature snapshot for dependence-plot reproducibility.
    np.savez_compressed(
        os.path.join(RESULTS_DIR, "shap_values_catboost.npz"),
        shap_values=shap_values,
        feature_names=np.array(tree.features),
        sample_idx=sample_idx,
    )
    pd.DataFrame(X_sample.values, columns=tree.features).to_csv(
        os.path.join(RESULTS_DIR, "shap_X_sample.csv"), index=False
    )
    print(importance_df.head(10).to_string(index=False))
    return importance_df


# ---------------------------------------------------------------------------
# Experiment 6: McNemar pairwise significance


def run_mcnemar():
    print("[exp6] McNemar pairwise significance on group-aware OOF predictions")
    oof = {}
    y_true = None
    for model_name in ALL_MODELS:
        path = os.path.join(RESULTS_DIR, f"oof_group_aware_{model_name}.csv")
        df = pd.read_csv(path)
        oof[model_name] = df["pred"].values.astype(int)
        if y_true is None:
            y_true = df["y_true"].values.astype(int)

    rows = []
    matrix_p = pd.DataFrame(index=list(ALL_MODELS), columns=list(ALL_MODELS), dtype=float)
    for a in ALL_MODELS:
        for b in ALL_MODELS:
            if a == b:
                matrix_p.loc[a, b] = 1.0
                continue
            corr_a = (oof[a] == y_true).astype(int)
            corr_b = (oof[b] == y_true).astype(int)
            b01 = int(((corr_a == 1) & (corr_b == 0)).sum())  # a right, b wrong
            b10 = int(((corr_a == 0) & (corr_b == 1)).sum())  # a wrong, b right
            n_disc = b01 + b10
            if n_disc == 0:
                p = 1.0
                stat = 0.0
            else:
                k = min(b01, b10)
                p = float(scipy_stats.binomtest(k, n_disc, 0.5, alternative="two-sided").pvalue)
                stat = float((abs(b01 - b10) - 1) ** 2 / max(n_disc, 1))  # continuity-corrected chi2
            matrix_p.loc[a, b] = p
            rows.append({
                "model_a": a, "model_b": b, "n": len(y_true),
                "a_right_b_wrong": b01, "a_wrong_b_right": b10,
                "chi2_continuity": stat, "p_value": p,
                "a_acc": float(accuracy_score(y_true, oof[a])),
                "b_acc": float(accuracy_score(y_true, oof[b])),
            })
            print(f"        [exp6] {a} vs {b}: a>b={b01} a<b={b10}  p={p:.4g}")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "mcnemar_pairwise.csv"), index=False)
    matrix_p.to_csv(os.path.join(RESULTS_DIR, "mcnemar_pvalue_matrix.csv"))
    return matrix_p


# ---------------------------------------------------------------------------
# Experiment 7: Calibration (Brier + reliability diagram)


def run_calibration():
    print("[exp7] calibration analysis on group-aware OOF probabilities")
    rows = []
    bin_records = []
    for model_name in ALL_MODELS:
        path = os.path.join(RESULTS_DIR, f"oof_group_aware_{model_name}.csv")
        df = pd.read_csv(path)
        y = df["y_true"].values.astype(int)
        p = df["proba"].values.astype(float)
        brier = float(brier_score_loss(y, p))
        # 10-bin reliability (uniform width)
        bins = np.linspace(0.0, 1.0, 11)
        bin_idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        for b in range(10):
            mask = bin_idx == b
            if not mask.any():
                continue
            bin_records.append({
                "model": model_name,
                "bin_lo": float(bins[b]),
                "bin_hi": float(bins[b + 1]),
                "count": int(mask.sum()),
                "mean_pred": float(p[mask].mean()),
                "frac_pos": float(y[mask].mean()),
            })
        rows.append({"model": model_name, "brier": brier, "n": len(y)})
        print(f"        [exp7] {model_name:<19s} brier={brier:.5f}")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "calibration_brier.csv"), index=False)
    pd.DataFrame(bin_records).to_csv(os.path.join(RESULTS_DIR, "calibration_bins.csv"), index=False)


# ---------------------------------------------------------------------------
# CLI


TASKS = {
    "group_cv": run_group_aware_cv,
    "eps_ablation": run_eps_ablation,
    "feature_ablation": run_feature_ablation,
    "shap": run_shap_analysis,
    "mcnemar": run_mcnemar,
    "calibration": run_calibration,
}


def main():
    parser = argparse.ArgumentParser(description="Run S1 experiments.")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS.keys()),
                        choices=list(TASKS.keys()))
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    global DATA_DIR
    DATA_DIR = args.data_dir

    ensure_dirs()
    print(f"results_dir = {RESULTS_DIR}")
    print(f"figures_dir = {FIGURES_DIR}")
    print(f"tables_dir  = {TABLES_DIR}")
    print(f"seed        = {SEED}")
    for name in args.tasks:
        print(f"\n=== {name} ===")
        TASKS[name]()


if __name__ == "__main__":
    main()
