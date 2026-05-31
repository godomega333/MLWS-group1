"""
E2-C: family_size + 5 spend zero-flags binary
=============================================
Goal: PassengerId 前缀同 family 行数 (family_size) + 5 个 'spending obs == 0' binary
zero_flag 特征是否提升 LB vs the team's 18-feature baseline.
The 18 base features and the team's main ablation tasks do not include any of
these 6 new features.

Modes:
- baseline_18 (team 18 features, equivalent to baseline_cb_18 anchor LB=0.81973)
- exp_24 (18 + family_size + 5 zero_flag)

zero_flag design: NaN -> -1 -> 0 (does not count as zero); obs == 0 -> 1; obs > 0 -> 0
Key: distinguish NaN from observed-zero, avoid fillna(median) confusion.
family_size: train+test concat, then PassengerId.str.split('_').str[0] groupby count.
Group-statistic over input features, not target-statistic (leak-free).

CV: 5-fold StratifiedKFold (SPLIT_RANDOM_STATE=42, dev metric).
LB: full retrain + Kaggle public-LB submission.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.cluster import DBSCAN
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from _common import DATA_DIR, RESULTS_DIR, kaggle_submission_notice

TRAIN_CSV = Path(DATA_DIR) / "train.csv"
TEST_CSV = Path(DATA_DIR) / "test.csv"

SEED = 78
N_SPLITS = 5
SPLIT_RANDOM_STATE = 42
CB_THRESHOLD = 0.4725

CB_PARAMS = dict(
    iterations=1358, learning_rate=0.0666, depth=5,
    l2_leaf_reg=8.036, bagging_temperature=0.424, border_count=211,
    thread_count=-1, allow_writing_files=False, random_seed=SEED, verbose=0,
)

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def build_features(mode):
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    y = train["Transported"].astype(int).values
    test_ids = test["PassengerId"]
    df = pd.concat([train.assign(is_train=1), test.assign(is_train=0)], ignore_index=True)

    if mode == "exp_24":
        df["FamilyId"] = df["PassengerId"].str.split("_").str[0]
        family_counts = df["FamilyId"].value_counts()
        df["family_size"] = df["FamilyId"].map(family_counts).astype(int)
        for col in SPEND_COLS:
            df[f"zero_flag_{col}"] = (df[col].fillna(-1) == 0).astype(int)
        n_obs_zero_total = sum((df[f"zero_flag_{c}"] == 1).sum() for c in SPEND_COLS)
        n_family_size_gt1 = (df["family_size"] > 1).sum()
        print(f"[exp_24] family_size>1 rows={n_family_size_gt1}/{len(df)}; total obs==0 flags={n_obs_zero_total}")

    df[["Deck", "Num", "Side"]] = df["Cabin"].str.split("/", expand=True)
    df["Num"] = pd.to_numeric(df["Num"], errors="coerce")

    for col in ["HomePlanet", "Side", "Deck", "Destination"]:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in ["Age"] + SPEND_COLS:
        df[col] = df[col].fillna(df[col].median())
    df["CryoSleep"] = df["CryoSleep"].fillna(False).astype(int)
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

    db = DBSCAN(eps=0.666, min_samples=2)
    df["Topology_Cluster"] = db.fit_predict(spatial_scaled)

    for col in ["Cabin_Region", "Age_Group", "Topology_Cluster"]:
        df[col] = df[col].fillna(-1).astype(int)

    base_18 = [
        "CryoSleep", "Cabin_Region", "Age_Group", "ShoppingMall", "Spend_per_Age",
        "Max_Spa_by_Destination", "LuxurySpend", "Deck", "Mean_Age_by_Deck",
        "RoomService", "Spa", "HomePlanet", "TotalSpend", "Max_Spa_by_Deck",
        "Mean_Spend_by_HomePlanet", "Mean_Spend_by_Deck", "Side", "Topology_Cluster",
    ]
    cat_18 = ["CryoSleep", "Cabin_Region", "Age_Group", "Deck", "HomePlanet", "Side", "Topology_Cluster"]

    if mode == "baseline_18":
        features = base_18
        cat_features = cat_18
    elif mode == "exp_24":
        new_features = ["family_size"] + [f"zero_flag_{c}" for c in SPEND_COLS]
        features = base_18 + new_features
        cat_features = cat_18  # new features are numeric/binary
    else:
        raise ValueError(mode)

    train_df = df[df["is_train"] == 1][features].reset_index(drop=True)
    test_df = df[df["is_train"] == 0][features].reset_index(drop=True)
    return train_df, test_df, y, test_ids, features, cat_features


def run_cv(X, y, cat_features, label):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SPLIT_RANDOM_STATE)
    fold_accs = []
    print(f"-- 5-fold StratifiedKFold CV ({label}) --")
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        model = CatBoostClassifier(**CB_PARAMS, cat_features=cat_features)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_va)[:, 1]
        preds = (proba >= CB_THRESHOLD).astype(int)
        acc = float((preds == y_va).mean())
        fold_accs.append(acc)
        print(f"fold {fold_idx + 1}/{N_SPLITS}: cv_acc={acc:.5f}")
    cv_mean = float(np.mean(fold_accs))
    cv_std = float(np.std(fold_accs))
    print(f"CV mean={cv_mean:.5f} std={cv_std:.5f}")
    return cv_mean, cv_std, fold_accs


def run_full_retrain(X_train, y, X_test, test_ids, cat_features, label):
    print("-- Full retrain + test inference --")
    model = CatBoostClassifier(**CB_PARAMS, cat_features=cat_features)
    model.fit(X_train, y)
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= CB_THRESHOLD).astype(bool)
    pred_csv = Path(RESULTS_DIR) / f"preds_e2_c_{label}.csv"
    pd.DataFrame({"PassengerId": test_ids, "Transported": preds}).to_csv(pred_csv, index=False)
    print(f"saved preds: {pred_csv}")
    return pred_csv


def run_one(label, mode):
    print("=" * 60)
    print(f"=== {label}  (mode={mode}) ===")
    print("=" * 60)
    t0 = time.time()
    X_train, X_test, y, test_ids, features, cat_features = build_features(mode)
    print(f"features ({len(features)}): {features}")
    print(f"cat_features: {cat_features}")
    cv_mean, cv_std, fold_accs = run_cv(X_train, y, cat_features, label)
    pred_csv = run_full_retrain(X_train, y, X_test, test_ids, cat_features, label)
    kaggle_submission_notice(str(pred_csv), label=label)
    wall = time.time() - t0
    print(f"wall_sec={wall:.1f}")
    return dict(label=label, mode=mode, n_features=len(features),
                cv_mean=cv_mean, cv_std=cv_std, fold_accs=fold_accs,
                pred_csv=str(pred_csv), wall_sec=wall)


def main():
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="*", default=["baseline_18", "exp_24"])
    args = ap.parse_args()

    results = {}
    for mode in args.modes:
        label = "baseline_cb_18" if mode == "baseline_18" else "exp_family_zeroflags_24"
        results[label] = run_one(label, mode)

    print("=" * 60)
    print("=== E2-C SUMMARY ===")
    print("=" * 60)
    for label, r in results.items():
        cv_str = f"{r['cv_mean']:.5f}+-{r['cv_std']:.5f}"
        print(f"{label:30s}: CV={cv_str}  n_feat={r['n_features']}  preds={r['pred_csv']}")
    print("Submit both prediction CSVs to Kaggle for public-LB scoring.")


    out_json = Path(RESULTS_DIR) / "e2_c_summary.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"summary saved: {out_json}")


if __name__ == "__main__":
    main()
