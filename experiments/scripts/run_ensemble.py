"""Optional S1 ensemble exploration.

Builds a simple soft-vote ensemble (mean of predicted probabilities across
LR / RF / XGB / CatBoost) at the group-aware OOF level, sweeps decision
thresholds, then refits each base model on the full train and emits a
candidate Kaggle submission CSV.

Outputs:
    experiments/results/ensemble_oof_summary.csv
    experiments/results/ensemble_threshold_sweep.csv
    experiments/results/submission_ensemble_mean.csv
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from _pipeline import (
    CB_PARAMS, CB_THRESHOLD, LR_PARAMS, RESULTS_DIR, RF_PARAMS, SEED,
    XGB_PARAMS, build_lr_frame, build_tree_frame,
)


ALL_MODELS = ("logistic_regression", "random_forest", "xgboost", "catboost")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Load OOF group-aware probabilities ---
    print("[ensemble] loading OOF probabilities")
    oof = {}
    y_true = None
    for m in ALL_MODELS:
        df = pd.read_csv(os.path.join(RESULTS_DIR, f"oof_group_aware_{m}.csv"))
        oof[m] = df["proba"].values.astype(float)
        if y_true is None:
            y_true = df["y_true"].values.astype(int)

    proba_mean = np.mean(np.column_stack([oof[m] for m in ALL_MODELS]), axis=1)
    sweep_rows = []
    for thr in np.arange(0.40, 0.55 + 1e-9, 0.005):
        pred = (proba_mean >= thr).astype(int)
        sweep_rows.append({"threshold": float(thr), "oof_acc": float(accuracy_score(y_true, pred))})
    sweep = pd.DataFrame(sweep_rows)
    best_row = sweep.loc[sweep["oof_acc"].idxmax()]
    sweep.to_csv(os.path.join(RESULTS_DIR, "ensemble_threshold_sweep.csv"), index=False)
    print(sweep.to_string(index=False))
    print(f"[ensemble] best threshold = {best_row['threshold']:.3f}, OOF acc = {best_row['oof_acc']:.5f}")

    base_summary = []
    for m in ALL_MODELS:
        thr = CB_THRESHOLD if m == "catboost" else 0.5
        pred = (oof[m] >= thr).astype(int)
        base_summary.append({
            "model": m,
            "oof_acc": float(accuracy_score(y_true, pred)),
            "oof_brier": float(brier_score_loss(y_true, oof[m])),
            "threshold": float(thr),
        })
    summary_df = pd.DataFrame(base_summary + [{
        "model": "ensemble_mean_thr0.5",
        "oof_acc": float(accuracy_score(y_true, (proba_mean >= 0.5).astype(int))),
        "oof_brier": float(brier_score_loss(y_true, proba_mean)),
        "threshold": 0.5,
    }, {
        "model": "ensemble_mean_thr_best",
        "oof_acc": float(best_row["oof_acc"]),
        "oof_brier": float(brier_score_loss(y_true, proba_mean)),
        "threshold": float(best_row["threshold"]),
    }])
    summary_df.to_csv(os.path.join(RESULTS_DIR, "ensemble_oof_summary.csv"), index=False)
    print(summary_df.round(5).to_string(index=False))

    # --- Refit each model on full train, average test-set probas ---
    print("[ensemble] refitting on full train and averaging test-set probabilities")
    tree = build_tree_frame()
    lr = build_lr_frame()
    y = tree.y_train

    test_probas = {}

    # LR
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(lr.train[lr.features].astype(float).values)
    X_te_s = scaler.transform(lr.test[lr.features].astype(float).values)
    model_lr = LogisticRegression(**LR_PARAMS)
    model_lr.fit(X_tr_s, y)
    test_probas["logistic_regression"] = model_lr.predict_proba(X_te_s)[:, 1]

    # RF
    model_rf = RandomForestClassifier(**RF_PARAMS)
    model_rf.fit(tree.train[tree.features], y)
    test_probas["random_forest"] = model_rf.predict_proba(tree.test[tree.features])[:, 1]

    # XGB
    model_xgb = XGBClassifier(**XGB_PARAMS)
    model_xgb.fit(tree.train[tree.features], y)
    test_probas["xgboost"] = model_xgb.predict_proba(tree.test[tree.features])[:, 1]

    # CatBoost
    model_cb = CatBoostClassifier(**CB_PARAMS, cat_features=tree.cat_features)
    model_cb.fit(tree.train[tree.features], y)
    test_probas["catboost"] = model_cb.predict_proba(tree.test[tree.features])[:, 1]

    test_mean = np.mean(np.column_stack([test_probas[m] for m in ALL_MODELS]), axis=1)
    thr = float(best_row["threshold"])
    test_pred = (test_mean >= thr).astype(bool)

    pd.DataFrame({
        "PassengerId": tree.test_ids,
        "Transported": test_pred,
    }).to_csv(os.path.join(RESULTS_DIR, "submission_ensemble_mean.csv"), index=False)
    pd.DataFrame({
        "PassengerId": tree.test_ids,
        **{f"proba_{m}": test_probas[m] for m in ALL_MODELS},
        "proba_mean": test_mean,
    }).to_csv(os.path.join(RESULTS_DIR, "ensemble_test_probas.csv"), index=False)
    print(f"[ensemble] wrote ensemble submission with threshold {thr:.3f}")


if __name__ == "__main__":
    main()
