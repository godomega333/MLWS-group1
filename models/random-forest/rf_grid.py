import argparse
import os
import random
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


DEFAULT_DATA_DIR = "spaceship-titanic-dataset"
DEFAULT_OUTPUT_DIR = os.path.join("models", "random-forest", "outputs")


def preprocess(data_dir):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))

    df = pd.concat([train.assign(is_train=1), test.assign(is_train=0)], ignore_index=True)
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['Num'] = pd.to_numeric(df['Num'], errors='coerce')

    for col in ['HomePlanet', 'Side', 'Deck', 'Destination']: df[col] = df[col].fillna(df[col].mode()[0])
    for col in ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']: df[col] = df[col].fillna(df[col].median())
    df['CryoSleep'] = df['CryoSleep'].fillna(False).astype(int)
    df['Deck_Code'] = df['Deck'].map({d: i for i, d in enumerate(sorted(df['Deck'].unique()))})
    df['Side_Code'] = df['Side'].map({'P': 0, 'S': 1})

    spatial_coords = df[['Deck_Code', 'Num', 'Side_Code']].fillna(0)
    spatial_scaled = StandardScaler().fit_transform(spatial_coords)

    df['TotalSpend'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']
    df['LuxurySpend'] = df['Spa'] + df['VRDeck'] + df['RoomService']
    df['Spend_per_Age'] = df['TotalSpend'] / (df['Age'] + 1)
    df['Age_Group'] = df['Age'] // 10
    df['Cabin_Region'] = df['Num'] // 300
    df['Mean_Spend_by_HomePlanet'] = df.groupby('HomePlanet')['TotalSpend'].transform('mean')
    df['Mean_Spend_by_Deck'] = df.groupby('Deck')['TotalSpend'].transform('mean')
    df['Mean_Age_by_Deck'] = df.groupby('Deck')['Age'].transform('mean')
    df['Max_Spa_by_Deck'] = df.groupby('Deck')['Spa'].transform('max')
    df['Max_Spa_by_Destination'] = df.groupby('Destination')['Spa'].transform('max')

    cat_cols_all = ['CryoSleep', 'Cabin_Region', 'Age_Group', 'Deck', 'HomePlanet', 'Side']
    for c in cat_cols_all: df[c] = df[c].astype(str).astype('category').cat.codes.astype(int)

    base_features = [
        'CryoSleep', 'Cabin_Region', 'Age_Group', 'ShoppingMall', 'Spend_per_Age',
        'Max_Spa_by_Destination', 'LuxurySpend', 'Deck', 'Mean_Age_by_Deck',
        'RoomService', 'Spa', 'HomePlanet', 'TotalSpend', 'Max_Spa_by_Deck',
        'Mean_Spend_by_HomePlanet', 'Mean_Spend_by_Deck', 'Side'
    ]

    y_train_full = train['Transported'].astype(int).values

    dbscan = DBSCAN(eps=0.666, min_samples=2)
    df['Topology_Cluster'] = dbscan.fit_predict(spatial_scaled)
    final_features = base_features + ['Topology_Cluster']

    train_df = df[df['is_train'] == 1].reset_index(drop=True)
    test_df = df[df['is_train'] == 0].reset_index(drop=True)

    return train_df, test_df, y_train_full, final_features, test['PassengerId']


def run_trial(trial_id, params, train_df, test_df, y_train_full, final_features, seed):
    rf_params = {
        'n_estimators': params['n_estimators'],
        'max_depth': params['max_depth'],
        'min_samples_split': params['min_samples_split'],
        'min_samples_leaf': params['min_samples_leaf'],
        'max_features': params['max_features'],
        'max_samples': params['max_samples'],
        'n_jobs': -1,
        'random_state': seed,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train_df))
    for train_idx, val_idx in skf.split(train_df, y_train_full):
        X_tr = train_df[final_features].iloc[train_idx]
        X_va = train_df[final_features].iloc[val_idx]
        model_cv = RandomForestClassifier(**rf_params)
        model_cv.fit(X_tr, y_train_full[train_idx])
        oof_preds[val_idx] = model_cv.predict(X_va)

    cv_acc = accuracy_score(y_train_full, oof_preds)

    model_full = RandomForestClassifier(**rf_params)
    model_full.fit(train_df[final_features], y_train_full)
    preds = [bool(p) for p in model_full.predict(test_df[final_features])]

    return {'Trial': trial_id, 'CV': cv_acc, 'params': params, 'preds': preds}


def parse_args():
    parser = argparse.ArgumentParser(description="RandomForest randomized grid search.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=78)
    parser.add_argument("--n-trials", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(">>> RandomForest Randomized Grid Search")
    print(f"    data_dir   = {args.data_dir}")
    print(f"    output_dir = {args.output_dir}")
    print(f"    seed       = {args.seed}")
    print(f"    n_trials   = {args.n_trials}")

    train_df, test_df, y_train_full, final_features, passenger_ids = preprocess(args.data_dir)

    search_space = []
    for _ in range(args.n_trials):
        search_space.append({
            'n_estimators': random.choice([500, 800, 1000, 1200, 1358, 1500, 2000]),
            'max_depth': random.choice([None, 5, 7, 10, 15, 20]),
            'min_samples_split': random.randint(2, 20),
            'min_samples_leaf': random.randint(1, 10),
            'max_features': random.choice(['sqrt', 'log2', 0.3, 0.5, 0.7, 1.0]),
            'max_samples': round(random.uniform(0.5, 1.0), 3),
        })

    results = []
    for i, params in enumerate(search_space):
        result = run_trial(i, params, train_df, test_df, y_train_full, final_features, args.seed)
        results.append(result)
        depth_str = str(params['max_depth']) if params['max_depth'] is not None else 'None'
        print(f"  Trial {i + 1:3d}/{args.n_trials} | CV={result['CV']:.5f} | Trees={params['n_estimators']} "
              f"| Depth={depth_str} | MaxFeat={params['max_features']} | MaxSamp={params['max_samples']:.3f}")

    results.sort(key=lambda x: x['CV'], reverse=True)

    rank10_idx = min(9, len(results) - 1)
    rank50_idx = min(49, len(results) - 1)
    targets = [("Rank1", results[0]), ("Rank10", results[rank10_idx]), ("Rank50", results[rank50_idx])]
    out_dir = args.output_dir

    print("\n" + "=" * 120)
    print("  RandomForest Grid Search Results")
    print("=" * 120)
    for rank_name, r in targets:
        p = r['params']
        depth_str = str(p['max_depth']) if p['max_depth'] is not None else 'None'
        print(f"  {rank_name}: CV={r['CV']:.5f} | Trees={p['n_estimators']} | Depth={depth_str} | "
              f"MinSplit={p['min_samples_split']} | MinLeaf={p['min_samples_leaf']} | "
              f"MaxFeat={p['max_features']} | MaxSamp={p['max_samples']}")
        file_name = os.path.join(out_dir,
            f"submission_rf_grid_{rank_name}_t{p['n_estimators']}_d{depth_str}"
            f"_mf{p['max_features']}_ms{p['max_samples']}.csv")
        pd.DataFrame({
            'PassengerId': passenger_ids,
            'Transported': r['preds']
        }).to_csv(file_name, index=False)
        print(f"  >>> Exported: {file_name}")
    print("=" * 120)


if __name__ == "__main__":
    main()
