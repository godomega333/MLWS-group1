from pathlib import Path

import modal


DATA_DIR = Path("spaceship-titanic-dataset")
REMOTE_DATA_DIR = Path("data")

app = modal.App("spaceship-catboost-grid-search")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgomp1")
    .pip_install("pandas", "numpy", "scikit-learn", "catboost")
    .add_local_file(str(DATA_DIR / "train.csv"), remote_path=str(REMOTE_DATA_DIR / "train.csv"))
    .add_local_file(str(DATA_DIR / "test.csv"), remote_path=str(REMOTE_DATA_DIR / "test.csv"))
)


@app.function(image=image, cpu=1.0, memory=2048, timeout=600)
def run_random_trial(trial_id, params):
    import warnings

    import numpy as np
    import pandas as pd
    from catboost import CatBoostClassifier
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    warnings.filterwarnings("ignore")

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

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

    y_train_full = train["Transported"].astype(int).values
    train_df = df[df["is_train"] == 1].reset_index(drop=True)
    test_df = df[df["is_train"] == 0].reset_index(drop=True)

    cb_params = {
        "iterations": 1357,
        "learning_rate": 0.0666,
        "depth": params["depth"],
        "l2_leaf_reg": params["l2_leaf_reg"],
        "bagging_temperature": params["bagging_temperature"],
        "border_count": params["border_count"],
        "thread_count": 1,
        "allow_writing_files": False,
        "verbose": False,
        "random_seed": 78,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train_df))
    for train_idx, val_idx in skf.split(train_df, y_train_full):
        X_tr = train_df[final_features].iloc[train_idx]
        X_va = train_df[final_features].iloc[val_idx]
        model_cv = CatBoostClassifier(**cb_params, cat_features=active_cat_cols)
        model_cv.fit(X_tr, y_train_full[train_idx])
        oof_preds[val_idx] = model_cv.predict(X_va)

    cv_acc = accuracy_score(y_train_full, oof_preds)

    model_full = CatBoostClassifier(**cb_params, cat_features=active_cat_cols)
    model_full.fit(train_df[final_features], y_train_full)
    preds = [bool(p) for p in model_full.predict(test_df[final_features])]

    return {"Trial": trial_id, "CV": cv_acc, "params": params, "preds": preds}


@app.local_entrypoint()
def main():
    import random

    import pandas as pd

    print(">>> CatBoost randomized grid search")
    print(">>> Fixed: iterations=1357, learning_rate=0.0666, eps=0.666")

    search_space = []
    for _ in range(100):
        search_space.append({
            "depth": random.randint(4, 8),
            "l2_leaf_reg": round(random.uniform(1.0, 10.0), 3),
            "bagging_temperature": round(random.uniform(0.1, 1.0), 3),
            "border_count": random.randint(100, 255),
        })

    inputs = [(i, params) for i, params in enumerate(search_space)]
    results = list(run_random_trial.starmap(inputs))
    results.sort(key=lambda x: x["CV"], reverse=True)

    rank10_idx = min(9, len(results) - 1)
    rank50_idx = min(49, len(results) - 1)
    targets = [("Rank1", results[0]), ("Rank10", results[rank10_idx]), ("Rank50", results[rank50_idx])]
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    print("\n" + "=" * 120)
    print("  CatBoost Grid Search Results")
    print("=" * 120)
    for rank_name, result in targets:
        params = result["params"]
        print(
            f"  {rank_name}: CV={result['CV']:.5f} | Depth={params['depth']} | "
            f"L2={params['l2_leaf_reg']} | Bagging={params['bagging_temperature']} | "
            f"Border={params['border_count']}"
        )
        file_name = (
            f"submission_catboost_grid_{rank_name}_d{params['depth']}"
            f"_l2{params['l2_leaf_reg']}_bg{params['bagging_temperature']}"
            f"_bd{params['border_count']}.csv"
        )
        pd.DataFrame({
            "PassengerId": test_df["PassengerId"],
            "Transported": result["preds"],
        }).to_csv(file_name, index=False)
        print(f"  >>> Exported: {file_name}")
    print("=" * 120)
