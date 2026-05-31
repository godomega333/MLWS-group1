"""
E2-G: LightGBM on team 18-feature pipeline (vs E2-A/C CatBoost baseline LB=0.81973).

Pipeline mirrors e2_a baseline byte-for-byte (team 18 features + cat.codes +
DBSCAN eps=0.666 Topology_Cluster). Only the model changes: CatBoost -> LightGBM.
LightGBM hyperparameters match team CB lr/n_est/lambda; others LightGBM default.

Outputs:
- results/preds_e2_g_lgb_18.csv  (PassengerId, Transported)
- results/proba_e2_g_lgb_18.csv  (PassengerId, proba)
- results/e2_g_summary.json

LB scoring: submit the preds CSV to the Kaggle Spaceship Titanic competition
page to obtain the public-leaderboard accuracy.
"""
from __future__ import annotations

import argparse
import json
import os
import random as py_random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.cluster import DBSCAN
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from _common import DATA_DIR, RESULTS_DIR, kaggle_submission_notice

RESULTS_DIR = Path(RESULTS_DIR)
DATA_DIR = Path(DATA_DIR)

SEED = 78
N_SPLITS = 5
SPLIT_RANDOM_STATE = 42
THRESHOLD = 0.5  # LightGBM default; team CB uses 0.4725 (CB-tuned, not transferable)

LGB_PARAMS = dict(
    objective="binary",
    metric="binary_error",
    learning_rate=0.0666,   # match team CB
    n_estimators=1358,      # match team CB
    num_leaves=31,          # LightGBM default
    reg_lambda=8.036,       # match team CB l2_leaf_reg
    random_state=SEED,
    verbosity=-1,
    n_jobs=-1,
)

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def build_frame(train, test):
    df = pd.concat([train.assign(is_train=1), test.assign(is_train=0)], ignore_index=True)
    df[["Deck", "Num", "Side"]] = df["Cabin"].str.split("/", expand=True)
    df["Num"] = pd.to_numeric(df["Num"], errors="coerce")

    for col in ["HomePlanet", "Side", "Deck", "Destination"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    df["CryoSleep"] = df["CryoSleep"].fillna(False).astype(int)

    for col in ["Age"] + SPEND_COLS:
        df[col] = df[col].fillna(df[col].median())

    df["Deck_Code"] = df["Deck"].map({d: i for i, d in enumerate(sorted(df["Deck"].unique()))})
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
    df["Topology_Cluster"] = dbscan.fit_predict(spatial_scaled).astype(int)
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
        m = LGBMClassifier(**LGB_PARAMS)
        m.fit(X_tr, y_tr, categorical_feature=cat_features)
        proba = m.predict_proba(X_va)[:, 1]
        preds = (proba >= THRESHOLD).astype(int)
        acc = float((preds == y_va).mean())
        accs.append(acc)
        print(f"    fold {fold + 1}/{N_SPLITS}: cv_acc={acc:.5f}")
    return float(np.mean(accs)), float(np.std(accs)), accs


def fit_full_predict(train_df, test_df, features, cat_features):
    y = train_df["Transported"].astype(int).values
    m = LGBMClassifier(**LGB_PARAMS)
    m.fit(train_df[features], y, categorical_feature=cat_features)
    return m.predict_proba(test_df[features])[:, 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cv", action="store_true")
    args = parser.parse_args()

    np.random.seed(SEED)
    py_random.seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    print(f">>> Loaded train={train.shape} test={test.shape}")

    label = "lgb_18"
    print("=" * 60)
    print(f"=== E2-G LightGBM 18-feature ({label}) ===")
    print("=" * 60)
    t0 = time.time()
    train_df, test_df, features, cat_features = build_frame(train, test)
    print(f"  features ({len(features)}): {features}")
    print(f"  cat_features: {cat_features}")

    if args.no_cv:
        cv_mean = cv_std = None
        cv_accs = []
    else:
        print(f"  -- {N_SPLITS}-fold StratifiedKFold CV --")
        cv_mean, cv_std, cv_accs = run_cv(train_df, features, cat_features)
        print(f"  CV mean={cv_mean:.5f} std={cv_std:.5f}")

    print("  -- Full retrain + test inference --")
    proba_test = fit_full_predict(train_df, test_df, features, cat_features)
    preds_test = (proba_test >= THRESHOLD).astype(bool)

    pred_csv = RESULTS_DIR / f"preds_e2_g_{label}.csv"
    proba_csv = RESULTS_DIR / f"proba_e2_g_{label}.csv"
    pd.DataFrame({"PassengerId": test_df["PassengerId"], "Transported": preds_test}).to_csv(pred_csv, index=False)
    pd.DataFrame({"PassengerId": test_df["PassengerId"], "proba": proba_test}).to_csv(proba_csv, index=False)
    print(f"  saved preds: {pred_csv}")

    kaggle_submission_notice(str(pred_csv), label=label)
    wall = time.time() - t0
    print(f"  wall_sec={wall:.1f}")

    print("\n" + "=" * 60)
    print("=== E2-G SUMMARY ===")
    print("=" * 60)
    cv_str = (f"{cv_mean:.5f}+-{cv_std:.5f}" if cv_mean is not None else "skipped")
    print(f"lgb_18:                CV={cv_str}  preds={pred_csv}")
    print("Submit prediction CSV to Kaggle for public-LB scoring.")

    summary = {
        "label": label,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "cv_accs": cv_accs,
        "pred_csv": str(pred_csv),
        "wall_sec": wall,
        "config": {
            "SEED": SEED,
            "N_SPLITS": N_SPLITS,
            "SPLIT_RANDOM_STATE": SPLIT_RANDOM_STATE,
            "THRESHOLD": THRESHOLD,
            "LGB_PARAMS": LGB_PARAMS,
            "model": "LightGBM",
            "n_features": len(features),
        },
    }
    out_path = RESULTS_DIR / "e2_g_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"summary saved: {out_path}")


if __name__ == "__main__":
    main()
