"""Shared feature engineering and model factories for S1 experiments.

Mirrors the per-family preprocessing in `models/*/*_comparison.py` and
`models/cat-boost/catboost_final.py`.
The tree-model pipeline (RF/XGB/CatBoost) and the CatBoost final-config use the same
feature set; the LR pipeline applies log-transforms and one-hot encoding.

All experiments in `experiments/scripts/run_experiments.py` import from this module.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


DEFAULT_DATA_DIR = "spaceship-titanic-dataset"
RESULTS_DIR = os.path.join("experiments", "results")
FIGURES_DIR = os.path.join("experiments", "figures")
TABLES_DIR = os.path.join("experiments", "tables")

SEED = 78


@dataclass(frozen=True)
class TreeFrame:
    train: pd.DataFrame
    test: pd.DataFrame
    features: list
    cat_features: list
    y_train: np.ndarray
    test_ids: pd.Series
    train_groups: np.ndarray  # cabin-cell group id per train row, NaN-cells pseudo-grouped


def _load_raw(data_dir: str):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    return train, test


def _make_groups(train_raw: pd.DataFrame) -> np.ndarray:
    """Group key = raw Cabin string. Missing cabin rows get a unique pseudo-group
    so they are never co-located with another row in any fold."""
    groups = train_raw["Cabin"].astype(object).copy()
    n_missing = groups.isna().sum()
    pseudo = [f"_NA_{i}" for i in range(n_missing)]
    groups.loc[groups.isna()] = pseudo
    return groups.astype(str).values


def build_tree_frame(
    data_dir: str = DEFAULT_DATA_DIR,
    eps: float = 0.666,
    min_samples: int = 2,
    drop_features: Optional[Iterable[str]] = None,
) -> TreeFrame:
    """Tree-model feature frame (CatBoost / RF / XGB).

    `drop_features` may include any of the canonical feature names defined in
    `_TREE_BASE_FEATURES` (plus `Topology_Cluster`); for the special bundle
    `__SPENDING__` we drop all raw spending and aggregated spending features.
    """
    train_raw, test_raw = _load_raw(data_dir)
    df = pd.concat([train_raw.assign(is_train=1), test_raw.assign(is_train=0)], ignore_index=True)
    df[["Deck", "Num", "Side"]] = df["Cabin"].str.split("/", expand=True)
    df["Num"] = pd.to_numeric(df["Num"], errors="coerce")

    for col in ["HomePlanet", "Side", "Deck", "Destination"]:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]:
        df[col] = df[col].fillna(df[col].median())
    df["CryoSleep"] = df["CryoSleep"].fillna(False).astype(int)
    df["Deck_Code"] = df["Deck"].map({d: i for i, d in enumerate(sorted(df["Deck"].unique()))})
    df["Side_Code"] = df["Side"].map({"P": 0, "S": 1})

    spatial_coords = df[["Deck_Code", "Num", "Side_Code"]].fillna(0)
    spatial_scaled = StandardScaler().fit_transform(spatial_coords)

    df["TotalSpend"] = df["RoomService"] + df["FoodCourt"] + df["ShoppingMall"] + df["Spa"] + df["VRDeck"]
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
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df["Topology_Cluster"] = dbscan.fit_predict(spatial_scaled)

    final_features = base_features + ["Topology_Cluster"]
    drop = set(drop_features or [])
    if "__SPENDING__" in drop:
        spending = {
            "RoomService", "ShoppingMall", "Spa", "TotalSpend", "LuxurySpend",
            "Spend_per_Age", "Max_Spa_by_Deck", "Max_Spa_by_Destination",
            "Mean_Spend_by_HomePlanet", "Mean_Spend_by_Deck",
        }
        drop |= spending
        drop.discard("__SPENDING__")
    final_features = [f for f in final_features if f not in drop]

    active_cat_cols = [c for c in cat_cols_all if c in final_features]
    if "Topology_Cluster" in final_features:
        active_cat_cols.append("Topology_Cluster")

    train_df = df[df["is_train"] == 1].reset_index(drop=True)
    test_df = df[df["is_train"] == 0].reset_index(drop=True)

    return TreeFrame(
        train=train_df,
        test=test_df,
        features=final_features,
        cat_features=active_cat_cols,
        y_train=train_raw["Transported"].astype(int).values,
        test_ids=test_raw["PassengerId"],
        train_groups=_make_groups(train_raw),
    )


@dataclass(frozen=True)
class LRFrame:
    train: pd.DataFrame
    test: pd.DataFrame
    features: list
    y_train: np.ndarray
    test_ids: pd.Series
    train_groups: np.ndarray


def build_lr_frame(data_dir: str = DEFAULT_DATA_DIR) -> LRFrame:
    train_raw, test_raw = _load_raw(data_dir)
    df = pd.concat([train_raw.assign(is_train=1), test_raw.assign(is_train=0)], ignore_index=True)
    df[["Deck", "Num", "Side"]] = df["Cabin"].str.split("/", expand=True)
    df["Num"] = pd.to_numeric(df["Num"], errors="coerce")

    for col in ["HomePlanet", "Side", "Deck", "Destination"]:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]:
        df[col] = df[col].fillna(df[col].median())
    df["CryoSleep"] = df["CryoSleep"].fillna(False).astype(int)
    df["Deck_Code"] = df["Deck"].map({d: i for i, d in enumerate(sorted(df["Deck"].unique()))})
    df["Side_Code"] = df["Side"].map({"P": 0, "S": 1})

    spatial_coords = df[["Deck_Code", "Num", "Side_Code"]].fillna(0)
    spatial_scaled = StandardScaler().fit_transform(spatial_coords)

    df["TotalSpend"] = df["RoomService"] + df["FoodCourt"] + df["ShoppingMall"] + df["Spa"] + df["VRDeck"]
    df["LuxurySpend"] = df["Spa"] + df["VRDeck"] + df["RoomService"]
    df["Spend_per_Age"] = df["TotalSpend"] / (df["Age"] + 1)
    df["Age_Group"] = df["Age"] // 10
    df["Cabin_Region"] = df["Num"] // 300
    df["Cabin_Region"] = df["Cabin_Region"].fillna(df["Cabin_Region"].median())
    df["Mean_Spend_by_HomePlanet"] = df.groupby("HomePlanet")["TotalSpend"].transform("mean")
    df["Mean_Spend_by_Deck"] = df.groupby("Deck")["TotalSpend"].transform("mean")
    df["Mean_Age_by_Deck"] = df.groupby("Deck")["Age"].transform("mean")
    df["Max_Spa_by_Deck"] = df.groupby("Deck")["Spa"].transform("max")
    df["Max_Spa_by_Destination"] = df.groupby("Destination")["Spa"].transform("max")

    df["Topology_Cluster"] = DBSCAN(eps=0.666, min_samples=2).fit_predict(spatial_scaled)

    log_cols = ["RoomService", "ShoppingMall", "Spa", "TotalSpend", "LuxurySpend", "Spend_per_Age"]
    for col in log_cols:
        df[col] = np.log1p(df[col])

    df["CryoSleep_x_TotalSpend"] = df["CryoSleep"] * df["TotalSpend"]
    df["CryoSleep_x_LuxurySpend"] = df["CryoSleep"] * df["LuxurySpend"]

    onehot_cols = ["Deck", "HomePlanet", "Side", "Topology_Cluster"]
    for col in onehot_cols:
        df[col] = df[col].astype(str)
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)

    base_numeric = [
        "CryoSleep", "Cabin_Region", "Age_Group", "ShoppingMall", "Spend_per_Age",
        "Max_Spa_by_Destination", "LuxurySpend", "Mean_Age_by_Deck",
        "RoomService", "Spa", "TotalSpend", "Max_Spa_by_Deck",
        "Mean_Spend_by_HomePlanet", "Mean_Spend_by_Deck",
        "CryoSleep_x_TotalSpend", "CryoSleep_x_LuxurySpend",
    ]
    onehot_features = [
        c for c in df.columns
        if any(c.startswith(p + "_") for p in ["Deck", "HomePlanet", "Side", "Topology_Cluster"])
    ]
    final_features = base_numeric + onehot_features

    train_df = df[df["is_train"] == 1].reset_index(drop=True)
    test_df = df[df["is_train"] == 0].reset_index(drop=True)

    return LRFrame(
        train=train_df,
        test=test_df,
        features=final_features,
        y_train=train_raw["Transported"].astype(int).values,
        test_ids=test_raw["PassengerId"],
        train_groups=_make_groups(train_raw),
    )


# Final-config hyperparameters per family (mirroring the comparison/Final scripts).

LR_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": SEED,
}

RF_PARAMS = {
    "n_estimators": 1358,
    "max_depth": 5,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "n_jobs": -1,
    "random_state": SEED,
}

XGB_PARAMS = {
    "n_estimators": 1358,
    "learning_rate": 0.0666,
    "max_depth": 5,
    "reg_lambda": 8.036,
    "subsample": 0.8,
    "max_bin": 211,
    "tree_method": "hist",
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "reg_alpha": 0,
    "n_jobs": 1,
    "verbosity": 0,
    "random_state": SEED,
}

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
