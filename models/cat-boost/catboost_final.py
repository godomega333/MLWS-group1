import argparse
import os
import random

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


DEFAULT_DATA_DIR = "spaceship-titanic-dataset"
DEFAULT_OUTPUT_DIR = os.path.join("models", "cat-boost", "outputs")


def build_feature_frame(train: pd.DataFrame, test: pd.DataFrame):
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

    dbscan = DBSCAN(eps=0.666, min_samples=2)
    df["Topology_Cluster"] = dbscan.fit_predict(spatial_scaled)
    final_features = base_features + ["Topology_Cluster"]
    active_cat_cols = [c for c in cat_cols_all if c in final_features] + ["Topology_Cluster"]

    train_df = df[df["is_train"] == 1].reset_index(drop=True)
    test_df = df[df["is_train"] == 0].reset_index(drop=True)
    return train_df, test_df, final_features, active_cat_cols


def fit_and_predict_proba(train_df, test_df, features, cat_features, seed):
    y_train_full = train_df["Transported"].astype(int).values

    cb_params = {
        "iterations": 1358,
        "learning_rate": 0.0666,
        "depth": 5,
        "l2_leaf_reg": 8.036,
        "bagging_temperature": 0.424,
        "border_count": 211,
        "thread_count": -1,
        "allow_writing_files": False,
        "verbose": False,
        "random_seed": seed,
    }
    print(f"    cb_params  = {cb_params}")

    model_full = CatBoostClassifier(**cb_params, cat_features=cat_features)
    model_full.fit(train_df[features], y_train_full)
    return model_full.predict_proba(test_df[features])[:, 1]


def parse_args():
    parser = argparse.ArgumentParser(description="CatBoost final-config inference.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="Directory containing train.csv and test.csv.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory for predictions and metrics.")
    parser.add_argument("--seed", type=int, default=78,
                        help="random_seed forwarded to CatBoost.")
    parser.add_argument("--threshold", type=float, default=0.4725,
                        help="Decision threshold applied to predicted probabilities.")
    parser.add_argument("--output-name", default="submission_catboost_final.csv",
                        help="Submission filename inside output-dir.")
    parser.add_argument("--proba-output-name", default="test_proba_catboost_final.csv",
                        help="Probability filename inside output-dir.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(">>> CatBoost final-config inference")
    print(f"    data_dir   = {args.data_dir}")
    print(f"    output_dir = {args.output_dir}")
    print(f"    seed       = {args.seed}")
    print(f"    threshold  = {args.threshold:.4f}")

    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))

    train_df, test_df, final_features, active_cat_cols = build_feature_frame(train, test)
    proba = fit_and_predict_proba(train_df, test_df, final_features, active_cat_cols, args.seed)

    preds = (proba >= args.threshold).astype(bool)
    out_path = os.path.join(args.output_dir, args.output_name)
    pd.DataFrame({"PassengerId": test_df["PassengerId"], "Transported": preds}).to_csv(out_path, index=False)
    pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "proba": proba,
    }).to_csv(os.path.join(args.output_dir, args.proba_output_name), index=False)

    print(f">>> Saved submission to: {out_path}")


if __name__ == "__main__":
    main()
