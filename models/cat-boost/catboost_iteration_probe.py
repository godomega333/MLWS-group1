from pathlib import Path

import modal


DATA_DIR = Path("spaceship-titanic-dataset")
REMOTE_DATA_DIR = Path("data")

app = modal.App("spaceship-catboost-iteration-probe")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgomp1")
    .pip_install("pandas", "numpy", "scikit-learn", "catboost")
    .add_local_file(str(DATA_DIR / "train.csv"), remote_path=str(REMOTE_DATA_DIR / "train.csv"))
    .add_local_file(str(DATA_DIR / "test.csv"), remote_path=str(REMOTE_DATA_DIR / "test.csv"))
)


@app.function(image=image, cpu=1.0, memory=2048)
def run_iteration_probe(iterations):
    import warnings

    import pandas as pd
    from catboost import CatBoostClassifier
    from sklearn.cluster import DBSCAN
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
        "iterations": iterations,
        "learning_rate": 0.0666,
        "depth": 5,
        "l2_leaf_reg": 8.036,
        "bagging_temperature": 0.424,
        "border_count": 211,
        "thread_count": 1,
        "allow_writing_files": False,
        "verbose": False,
        "random_seed": 78,
    }

    model_full = CatBoostClassifier(**cb_params, cat_features=active_cat_cols)
    model_full.fit(train_df[final_features], y_train_full)
    preds = [bool(p) for p in model_full.predict(test_df[final_features])]

    return iterations, preds


@app.local_entrypoint()
def main():
    import pandas as pd

    test_iterations = [1300, 1333, 1350, 1356, 1357, 1358, 1359, 1366, 1388, 1400]
    print(f">>> CatBoost iteration probe: {test_iterations}")

    results = list(run_iteration_probe.map(test_iterations))
    results.sort(key=lambda x: x[0])

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    print("\n" + "=" * 80)
    for iterations, preds in results:
        file_name = f"submission_catboost_iter_{iterations}.csv"
        pd.DataFrame({
            "PassengerId": test_df["PassengerId"],
            "Transported": preds,
        }).to_csv(file_name, index=False)
        print(f"  >>> Exported: {file_name}")
    print("=" * 80)
