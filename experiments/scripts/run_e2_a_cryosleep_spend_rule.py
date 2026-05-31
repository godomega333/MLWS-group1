"""E2-A: CryoSleep spending-rule imputation.

Compares two CryoSleep imputation modes on the 18-feature CatBoost pipeline:
  baseline_cb_18:      df["CryoSleep"].fillna(False)
  exp_cryo_spend_rule: rows with all 5 spend cols == 0 (and CryoSleep NaN) -> True

Pipeline mirrors the team baseline. Only build_frame's CryoSleep imputation
differs between the two modes.

Outputs:
- results/preds_e2_a_{label}.csv          (PassengerId, Transported)
- results/proba_e2_a_{label}.csv          (PassengerId, proba)
- results/e2_a_summary.json               (CV summary, plus Kaggle anchors from paper)

LB scoring: submit each preds CSV to the Kaggle Spaceship Titanic competition
page to obtain the public-leaderboard accuracy.
"""
from __future__ import annotations

import argparse
import json
import os
import random as py_random
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.cluster import DBSCAN
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from _common import DATA_DIR, RESULTS_DIR, kaggle_submission_notice

SEED = 78
N_SPLITS = 5
SPLIT_RANDOM_STATE = 42
CB_PARAMS = {
    "iterations": 1358,
    "learning_rate": 0.0666,
    "depth": 5,
    "l2_leaf_reg": 8.036,
    "bagging_temperature": 0.424,
    "border_count": 211,
    "thread_count": -1,
    "allow_writing_files": False,
    "verbose": False,
    "random_seed": SEED,
}
CB_THRESHOLD = 0.4725
SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def build_frame(train, test, cryo_mode):
    df = pd.concat(
        [train.assign(is_train=1), test.assign(is_train=0)], ignore_index=True
    )
    df[["Deck", "Num", "Side"]] = df["Cabin"].str.split("/", expand=True)
    df["Num"] = pd.to_numeric(df["Num"], errors="coerce")

    for col in ["HomePlanet", "Side", "Deck", "Destination"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    if cryo_mode == "fillna_false":
        n_nan = int(df["CryoSleep"].isna().sum())
        df["CryoSleep"] = df["CryoSleep"].fillna(False).astype(int)
        print(f"  [CryoSleep fillna_false] {n_nan} NaN -> False (default)")
    elif cryo_mode == "spend_rule":
        spend_known_zero = (df[SPEND_COLS] == 0).all(axis=1)
        nan_cryo = df["CryoSleep"].isna()
        n_nan = int(nan_cryo.sum())
        n_to_true = int((nan_cryo & spend_known_zero).sum())
        n_to_false = n_nan - n_to_true
        df.loc[nan_cryo & spend_known_zero, "CryoSleep"] = True
        df["CryoSleep"] = df["CryoSleep"].fillna(False).astype(int)
        print(
            f"  [CryoSleep spend_rule] NaN={n_nan}  -> True={n_to_true}  -> False={n_to_false}"
        )
    else:
        raise ValueError(f"unknown cryo_mode: {cryo_mode}")

    for col in ["Age"] + SPEND_COLS:
        df[col] = df[col].fillna(df[col].median())

    df["Deck_Code"] = df["Deck"].map(
        {d: i for i, d in enumerate(sorted(df["Deck"].unique()))}
    )
    df["Side_Code"] = df["Side"].map({"P": 0, "S": 1})

    spatial = df[["Deck_Code", "Num", "Side_Code"]].fillna(0)
    spatial_scaled = StandardScaler().fit_transform(spatial)

    df["TotalSpend"] = df[SPEND_COLS].sum(axis=1)
    df["LuxurySpend"] = df["Spa"] + df["VRDeck"] + df["RoomService"]
    df["Spend_per_Age"] = df["TotalSpend"] / (df["Age"] + 1)
    df["Age_Group"] = df["Age"] // 10
    df["Cabin_Region"] = df["Num"] // 300
    df["Mean_Spend_by_HomePlanet"] = df.groupby("HomePlanet")["TotalSpend"].transform("mean")
    df["Mean_Spend_by_Deck"] = df.groupby("Deck")["TotalSpend"].transform("mean")
    df["Mean_Age_by_Deck"] = df.groupby("Deck")["Age"].transform("mean")
    df["Max_Spa_by_Deck"] = df.groupby("Deck")["Spa"].transform("max")
    df["Max_Spa_by_Destination"] = df.groupby("Destination")["Spa"].transform("max")

    cat_cols_all = ["CryoSleep", "Cabin_Region", "Age_Group", "Deck", "HomePlanet", "Side"]
    for col in cat_cols_all:
        df[col] = df[col].astype(str).astype("category").cat.codes.astype(int)

    base_features = [
        "CryoSleep", "Cabin_Region", "Age_Group", "ShoppingMall", "Spend_per_Age",
        "Max_Spa_by_Destination", "LuxurySpend", "Deck", "Mean_Age_by_Deck",
        "RoomService", "Spa", "HomePlanet", "TotalSpend", "Max_Spa_by_Deck",
        "Mean_Spend_by_HomePlanet", "Mean_Spend_by_Deck", "Side",
    ]
    dbscan = DBSCAN(eps=0.666, min_samples=2)
    df["Topology_Cluster"] = dbscan.fit_predict(spatial_scaled)
    final_features = base_features + ["Topology_Cluster"]
    active_cat_cols = [c for c in cat_cols_all if c in final_features] + ["Topology_Cluster"]

    train_df = df[df["is_train"] == 1].reset_index(drop=True)
    test_df = df[df["is_train"] == 0].reset_index(drop=True)
    return train_df, test_df, final_features, active_cat_cols


def run_cv(train_df, features, cat_features):
    y = train_df["Transported"].astype(int).values
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SPLIT_RANDOM_STATE)
    accs = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df[features], y)):
        X_tr = train_df.iloc[tr_idx][features]
        X_va = train_df.iloc[va_idx][features]
        y_tr, y_va = y[tr_idx], y[va_idx]
        m = CatBoostClassifier(**CB_PARAMS, cat_features=cat_features)
        m.fit(X_tr, y_tr)
        proba = m.predict_proba(X_va)[:, 1]
        preds = (proba >= CB_THRESHOLD).astype(int)
        acc = float((preds == y_va).mean())
        accs.append(acc)
        print(f"    fold {fold + 1}/{N_SPLITS}: cv_acc={acc:.5f}")
    return float(np.mean(accs)), float(np.std(accs)), accs


def fit_full_predict(train_df, test_df, features, cat_features):
    y = train_df["Transported"].astype(int).values
    m = CatBoostClassifier(**CB_PARAMS, cat_features=cat_features)
    m.fit(train_df[features], y)
    return m.predict_proba(test_df[features])[:, 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cv", action="store_true")
    args = parser.parse_args()

    np.random.seed(SEED)
    py_random.seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f">>> Loaded train={train.shape} test={test.shape}")

    results = {}
    for cryo_mode, label in [
        ("fillna_false", "baseline_cb_18"),
        ("spend_rule", "exp_cryo_spend_rule"),
    ]:
        print("\n" + "=" * 60)
        print(f"=== {label}  (cryo_mode={cryo_mode}) ===")
        print("=" * 60)
        t0 = time.time()
        train_df, test_df, features, cat_features = build_frame(
            train.copy(), test.copy(), cryo_mode
        )
        print(f"  features ({len(features)}): {features}")
        print(f"  cat_features: {cat_features}")
        cv_mean = cv_std = None
        cv_accs = []
        if not args.no_cv:
            print(f"  -- Running {N_SPLITS}-fold StratifiedKFold CV --")
            cv_mean, cv_std, cv_accs = run_cv(train_df, features, cat_features)
            print(f"  CV mean={cv_mean:.5f} std={cv_std:.5f}")
        print("  -- Full retrain + test inference --")
        proba_test = fit_full_predict(train_df, test_df, features, cat_features)
        preds_test = (proba_test >= CB_THRESHOLD).astype(bool)
        pred_csv = os.path.join(RESULTS_DIR, f"preds_e2_a_{label}.csv")
        proba_csv = os.path.join(RESULTS_DIR, f"proba_e2_a_{label}.csv")
        pd.DataFrame({
            "PassengerId": test_df["PassengerId"],
            "Transported": preds_test,
        }).to_csv(pred_csv, index=False)
        pd.DataFrame({
            "PassengerId": test_df["PassengerId"],
            "proba": proba_test,
        }).to_csv(proba_csv, index=False)
        print(f"  saved preds: {pred_csv}")
        kaggle_submission_notice(pred_csv, label=label)
        wall = time.time() - t0
        results[label] = {
            "cryo_mode": cryo_mode,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "cv_accs": cv_accs,
            "pred_csv": pred_csv,
            "wall_sec": wall,
        }
        print(f"  wall_sec={wall:.1f}")

    print("\n" + "=" * 60)
    print("=== E2-A SUMMARY ===")
    print("=" * 60)
    base = results["baseline_cb_18"]
    exp = results["exp_cryo_spend_rule"]
    b_cv = (f"{base['cv_mean']:.5f}+-{base['cv_std']:.5f}" if base["cv_mean"] is not None else "skipped")
    e_cv = (f"{exp['cv_mean']:.5f}+-{exp['cv_std']:.5f}" if exp["cv_mean"] is not None else "skipped")
    print(f"baseline_cb_18:        CV={b_cv}  preds={base['pred_csv']}")
    print(f"exp_cryo_spend_rule:   CV={e_cv}  preds={exp['pred_csv']}")
    print("Submit both prediction CSVs to Kaggle for public-LB scoring.")

    summary = {
        "baseline_cb_18": base,
        "exp_cryo_spend_rule": exp,
        "config": {
            "SEED": SEED,
            "N_SPLITS": N_SPLITS,
            "SPLIT_RANDOM_STATE": SPLIT_RANDOM_STATE,
            "CB_PARAMS": CB_PARAMS,
            "CB_THRESHOLD": CB_THRESHOLD,
        },
    }
    out_path = os.path.join(RESULTS_DIR, "e2_a_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"summary saved: {out_path}")


if __name__ == "__main__":
    main()
