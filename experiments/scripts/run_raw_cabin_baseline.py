"""E1: raw Cabin baseline counterexample for the high-cardinality cabin
encoding hypothesis.

Replaces Cabin_Region (Num // 300, ~30 regions) with raw Cabin string
(~6560 unique cells) as a high-cardinality categorical. Runs CatBoost
under group-aware StratifiedGroupKFold and stratified StratifiedKFold;
reports both gap and full-train inference for Kaggle public-LB
submission.

Hypothesis: under raw Cabin, stratified CV inflates (same-cell train/val
rows leak ordered-TS signal) while group-aware CV degrades (no overlap),
producing a measurable gap.

The splitter ALWAYS uses raw Cabin as the group key; only the feature
representation differs between modes.
"""
from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from _common import DATA_DIR, RESULTS_DIR, kaggle_submission_notice  # noqa: E402

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


def make_groups(train_raw):
    g = train_raw["Cabin"].astype(object).copy()
    n_missing = g.isna().sum()
    pseudo = [f"_NA_{i}" for i in range(n_missing)]
    g.loc[g.isna()] = pseudo
    return g.astype(str).values


def build_frame(data_dir, mode):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    df = pd.concat(
        [train.assign(is_train=1), test.assign(is_train=0)], ignore_index=True
    )
    df[["Deck", "Num", "Side"]] = df["Cabin"].str.split("/", expand=True)
    df["Num"] = pd.to_numeric(df["Num"], errors="coerce")

    for col in ["HomePlanet", "Side", "Deck", "Destination"]:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]:
        df[col] = df[col].fillna(df[col].median())
    df["CryoSleep"] = df["CryoSleep"].fillna(False).astype(int)
    df["Deck_Code"] = df["Deck"].map(
        {d: i for i, d in enumerate(sorted(df["Deck"].unique()))}
    )
    df["Side_Code"] = df["Side"].map({"P": 0, "S": 1})

    spatial = df[["Deck_Code", "Num", "Side_Code"]].fillna(0)
    spatial_scaled = StandardScaler().fit_transform(spatial)

    df["TotalSpend"] = df["RoomService"] + df["FoodCourt"] + df["ShoppingMall"] + df["Spa"] + df["VRDeck"]
    df["LuxurySpend"] = df["Spa"] + df["VRDeck"] + df["RoomService"]
    df["Spend_per_Age"] = df["TotalSpend"] / (df["Age"] + 1)
    df["Age_Group"] = df["Age"] // 10
    df["Mean_Spend_by_HomePlanet"] = df.groupby("HomePlanet")["TotalSpend"].transform("mean")
    df["Mean_Spend_by_Deck"] = df.groupby("Deck")["TotalSpend"].transform("mean")
    df["Mean_Age_by_Deck"] = df.groupby("Deck")["Age"].transform("mean")
    df["Max_Spa_by_Deck"] = df.groupby("Deck")["Spa"].transform("max")
    df["Max_Spa_by_Destination"] = df.groupby("Destination")["Spa"].transform("max")

    if mode == "cabin_region":
        df["Cabin_Region"] = df["Num"] // 300
        cabin_feature = "Cabin_Region"
    elif mode == "raw_cabin":
        df["Cabin_Raw"] = df["Cabin"].fillna("__NA__").astype(str)
        cabin_feature = "Cabin_Raw"
    else:
        raise ValueError(f"unknown mode: {mode}")

    cat_cols_all = ["CryoSleep", cabin_feature, "Age_Group", "Deck", "HomePlanet", "Side"]
    for col in cat_cols_all:
        df[col] = df[col].astype(str).astype("category").cat.codes.astype(int)

    base_features = [
        "CryoSleep", cabin_feature, "Age_Group", "ShoppingMall", "Spend_per_Age",
        "Max_Spa_by_Destination", "LuxurySpend", "Deck", "Mean_Age_by_Deck",
        "RoomService", "Spa", "HomePlanet", "TotalSpend", "Max_Spa_by_Deck",
        "Mean_Spend_by_HomePlanet", "Mean_Spend_by_Deck", "Side",
    ]
    df["Topology_Cluster"] = DBSCAN(eps=0.666, min_samples=2).fit_predict(spatial_scaled)
    final_features = base_features + ["Topology_Cluster"]
    active_cat = [c for c in cat_cols_all if c in final_features] + ["Topology_Cluster"]

    train_df = df[df["is_train"] == 1].reset_index(drop=True)
    test_df = df[df["is_train"] == 0].reset_index(drop=True)
    n_unique_cabin = df[cabin_feature].nunique()
    return train_df, test_df, final_features, active_cat, n_unique_cabin


def run_cv(train_df, features, cat_features, y, splits, label):
    fold_accs = []
    for fold_idx, (tr, va) in enumerate(splits):
        t0 = time.perf_counter()
        model = CatBoostClassifier(**CB_PARAMS, cat_features=cat_features)
        model.fit(train_df[features].iloc[tr], y[tr])
        proba = model.predict_proba(train_df[features].iloc[va])[:, 1]
        pred = (proba >= CB_THRESHOLD).astype(int)
        acc = accuracy_score(y[va], pred)
        fold_accs.append(acc)
        print(f"  [{label}] fold {fold_idx + 1}/{N_SPLITS} acc={acc:.5f} ({time.perf_counter()-t0:.1f}s)")
    arr = np.array(fold_accs)
    sem = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return arr, float(arr.mean()), float(1.96 * sem)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["both", "cabin_region", "raw_cabin"], default="both")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    groups = make_groups(train_raw)
    print(f"groups: {len(np.unique(groups))} unique cabin keys (train)")

    modes = ["cabin_region", "raw_cabin"] if args.mode == "both" else [args.mode]
    rows = []
    full_pred_paths = {}

    for mode in modes:
        print(f"\n=== mode = {mode} ===")
        train_df, test_df, features, cat_features, n_unique = build_frame(DATA_DIR, mode)
        y = train_df["Transported"].astype(int).values
        print(f"  features ({len(features)}): {features}")
        print(f"  cat_features: {cat_features}")
        print(f"  unique levels of cabin feature ({mode}): {n_unique}")

        sgkf_splits = list(
            StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                 random_state=SPLIT_RANDOM_STATE).split(train_df, y, groups)
        )
        skf_splits = list(
            StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                            random_state=SPLIT_RANDOM_STATE).split(train_df, y)
        )

        ga_arr, ga_mean, ga_ci = run_cv(train_df, features, cat_features, y, sgkf_splits, f"{mode}/group-aware")
        st_arr, st_mean, st_ci = run_cv(train_df, features, cat_features, y, skf_splits, f"{mode}/stratified")

        gap = st_mean - ga_mean
        print(f"  >> [{mode}] group-aware mean = {ga_mean:.5f} +/- {ga_ci:.5f}")
        print(f"  >> [{mode}] stratified  mean = {st_mean:.5f} +/- {st_ci:.5f}")
        print(f"  >> [{mode}] gap (strat-group) = {gap:+.5f}")

        if mode == "cabin_region":
            exp_ga, exp_st = 0.81215, 0.81272
            print(f"  [paper anchor] GA: {ga_mean:.5f} vs {exp_ga} (delta={ga_mean-exp_ga:+.5f})")
            print(f"  [paper anchor] ST: {st_mean:.5f} vs {exp_st} (delta={st_mean-exp_st:+.5f})")

        rows.append({
            "mode": mode,
            "n_features": len(features),
            "n_unique_cabin_levels": n_unique,
            "group_aware_mean": ga_mean,
            "group_aware_ci95_half": ga_ci,
            "stratified_mean": st_mean,
            "stratified_ci95_half": st_ci,
            "gap_strat_minus_group": gap,
        })

        print(f"  [{mode}] full-train fit + test inference...")
        t0 = time.perf_counter()
        model_full = CatBoostClassifier(**CB_PARAMS, cat_features=cat_features)
        model_full.fit(train_df[features], y)
        proba_test = model_full.predict_proba(test_df[features])[:, 1]
        preds_test = (proba_test >= CB_THRESHOLD).astype(bool)
        out_path = os.path.join(RESULTS_DIR, f"preds_e1_{mode}.csv")
        pd.DataFrame({"PassengerId": test_df["PassengerId"], "Transported": preds_test}).to_csv(
            out_path, index=False)
        full_pred_paths[mode] = out_path
        print(f"  done in {time.perf_counter()-t0:.1f}s -> {out_path}")

    df_out = pd.DataFrame(rows)
    summary_path = os.path.join(RESULTS_DIR, "raw_cabin_baseline.csv")
    df_out.to_csv(summary_path, index=False)
    print(f"\n=== summary saved: {summary_path} ===")
    print(df_out.to_string(index=False))

    print("\n=== Kaggle public-LB submission ===")
    for mode, path in full_pred_paths.items():
        kaggle_submission_notice(path, label=f"e1_{mode}")

    print("\nE1 done.")


if __name__ == "__main__":
    main()
