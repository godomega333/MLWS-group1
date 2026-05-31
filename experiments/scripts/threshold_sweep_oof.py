from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.cluster import DBSCAN
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path("spaceship-titanic-dataset")
RESULTS_DIR = Path("experiments") / "results"


def build_feature_frame() -> tuple[pd.DataFrame, list[str], list[str], np.ndarray]:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    df = pd.concat([train.assign(is_train=1), test.assign(is_train=0)], ignore_index=True)
    df[["Deck", "Num", "Side"]] = df["Cabin"].str.split("/", expand=True)
    df["Num"] = pd.to_numeric(df["Num"], errors="coerce")

    for col in ["HomePlanet", "Side", "Deck", "Destination"]:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]:
        df[col] = df[col].fillna(df[col].median())

    df["CryoSleep"] = df["CryoSleep"].fillna(False).astype(int)
    df["Deck_Code"] = df["Deck"].map({d: i for i, d in enumerate(sorted(df["Deck"].unique()))})
    df["Side_Code"] = df["Side"].map({"P": 0, "S": 1})

    spatial_scaled = StandardScaler().fit_transform(df[["Deck_Code", "Num", "Side_Code"]].fillna(0))

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

    df["Topology_Cluster"] = DBSCAN(eps=0.666, min_samples=2).fit_predict(spatial_scaled)

    base_features = [
        "CryoSleep",
        "Cabin_Region",
        "Age_Group",
        "ShoppingMall",
        "Spend_per_Age",
        "Max_Spa_by_Destination",
        "LuxurySpend",
        "Deck",
        "Mean_Age_by_Deck",
        "RoomService",
        "Spa",
        "HomePlanet",
        "TotalSpend",
        "Max_Spa_by_Deck",
        "Mean_Spend_by_HomePlanet",
        "Mean_Spend_by_Deck",
        "Side",
    ]
    final_features = base_features + ["Topology_Cluster"]
    active_cat_cols = [c for c in cat_cols_all if c in final_features] + ["Topology_Cluster"]

    train_df = df[df["is_train"] == 1].reset_index(drop=True)
    y = train_df["Transported"].astype(int).values
    groups = train_df["Cabin"].astype(object).copy()
    missing = groups.isna()
    groups.loc[missing] = [f"_NA_{i}" for i in range(int(missing.sum()))]

    return train_df, final_features, active_cat_cols, y, groups.astype(str).values


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_df, final_features, active_cat_cols, y, groups = build_feature_frame()

    cb_params = {
        "iterations": 1358,
        "learning_rate": 0.0666,
        "depth": 5,
        "l2_leaf_reg": 8.036,
        "bagging_temperature": 0.424,
        "border_count": 211,
        "random_seed": 78,
        "thread_count": -1,
        "allow_writing_files": False,
        "verbose": False,
    }

    oof_proba = np.zeros(len(train_df), dtype=float)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(train_df, y, groups)):
        model = CatBoostClassifier(**cb_params, cat_features=active_cat_cols)
        model.fit(train_df[final_features].iloc[train_idx], y[train_idx])
        oof_proba[val_idx] = model.predict_proba(train_df[final_features].iloc[val_idx])[:, 1]
        fold_acc = np.mean((oof_proba[val_idx] >= 0.5) == y[val_idx])
        print(f"fold {fold_idx + 1}/5 done, fold acc @0.5 = {fold_acc:.6f}")

    print(f"\nOOF acc @ default 0.5 = {np.mean((oof_proba >= 0.5) == y):.6f}")

    thresholds = np.arange(0.40, 0.55001, 0.0005)
    oof_acc = np.array([np.mean((oof_proba >= threshold) == y) for threshold in thresholds])
    output_path = RESULTS_DIR / "threshold_sweep_oof.csv"
    pd.DataFrame({"threshold": thresholds, "oof_acc": oof_acc}).to_csv(output_path, index=False)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
