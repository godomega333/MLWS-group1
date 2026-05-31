import argparse
import os
import random
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


DEFAULT_DATA_DIR = "spaceship-titanic-dataset"
DEFAULT_OUTPUT_DIR = os.path.join("models", "logistic-regression", "outputs")


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
    df['Cabin_Region'] = df['Cabin_Region'].fillna(df['Cabin_Region'].median())
    df['Mean_Spend_by_HomePlanet'] = df.groupby('HomePlanet')['TotalSpend'].transform('mean')
    df['Mean_Spend_by_Deck'] = df.groupby('Deck')['TotalSpend'].transform('mean')
    df['Mean_Age_by_Deck'] = df.groupby('Deck')['Age'].transform('mean')
    df['Max_Spa_by_Deck'] = df.groupby('Deck')['Spa'].transform('max')
    df['Max_Spa_by_Destination'] = df.groupby('Destination')['Spa'].transform('max')

    # DBSCAN topology clustering
    dbscan = DBSCAN(eps=0.666, min_samples=2)
    df['Topology_Cluster'] = dbscan.fit_predict(spatial_scaled)

    # --- LR-specific adaptation layer ---

    # 1. Log-transform skewed spending features
    log_cols = ['RoomService', 'ShoppingMall', 'Spa', 'TotalSpend', 'LuxurySpend', 'Spend_per_Age']
    for col in log_cols:
        df[col] = np.log1p(df[col])

    # 2. Interaction features
    df['CryoSleep_x_TotalSpend'] = df['CryoSleep'] * df['TotalSpend']
    df['CryoSleep_x_LuxurySpend'] = df['CryoSleep'] * df['LuxurySpend']

    # 3. One-Hot encode non-ordinal categoricals
    onehot_cols = ['Deck', 'HomePlanet', 'Side', 'Topology_Cluster']
    for col in onehot_cols:
        df[col] = df[col].astype(str)
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)

    # Build final feature list
    base_numeric = [
        'CryoSleep', 'Cabin_Region', 'Age_Group', 'ShoppingMall', 'Spend_per_Age',
        'Max_Spa_by_Destination', 'LuxurySpend', 'Mean_Age_by_Deck',
        'RoomService', 'Spa', 'TotalSpend', 'Max_Spa_by_Deck',
        'Mean_Spend_by_HomePlanet', 'Mean_Spend_by_Deck',
        'CryoSleep_x_TotalSpend', 'CryoSleep_x_LuxurySpend',
    ]
    onehot_features = [c for c in df.columns if any(c.startswith(p + '_') for p in ['Deck', 'HomePlanet', 'Side', 'Topology_Cluster'])]
    final_features = base_numeric + onehot_features

    y_train_full = train['Transported'].astype(int).values

    train_df = df[df['is_train'] == 1].reset_index(drop=True)
    test_df = df[df['is_train'] == 0].reset_index(drop=True)

    return train_df, test_df, y_train_full, final_features, test['PassengerId']


def run_trial(trial_id, params, train_df, test_df, y_train_full, final_features, seed):
    solver = 'liblinear' if params['penalty'] == 'l1' else 'lbfgs'

    lr_params = {
        'C': params['C'],
        'penalty': params['penalty'],
        'solver': solver,
        'max_iter': 2000,
        'random_state': seed,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train_df))
    scaler = StandardScaler()

    for train_idx, val_idx in skf.split(train_df, y_train_full):
        X_tr = train_df[final_features].iloc[train_idx]
        X_va = train_df[final_features].iloc[val_idx]

        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)

        model_cv = LogisticRegression(**lr_params)
        model_cv.fit(X_tr_scaled, y_train_full[train_idx])
        oof_preds[val_idx] = model_cv.predict(X_va_scaled)

    cv_acc = accuracy_score(y_train_full, oof_preds)

    X_train_all = scaler.fit_transform(train_df[final_features])
    X_test_all = scaler.transform(test_df[final_features])

    model_full = LogisticRegression(**lr_params)
    model_full.fit(X_train_all, y_train_full)
    preds = [bool(p) for p in model_full.predict(X_test_all)]

    return {'Trial': trial_id, 'CV': cv_acc, 'params': params, 'preds': preds}


def parse_args():
    parser = argparse.ArgumentParser(description="LR randomized grid search.")
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

    print(">>> LogisticRegression Randomized Grid Search")
    print(f"    data_dir   = {args.data_dir}")
    print(f"    output_dir = {args.output_dir}")
    print(f"    seed       = {args.seed}")
    print(f"    n_trials   = {args.n_trials}")

    train_df, test_df, y_train_full, final_features, passenger_ids = preprocess(args.data_dir)

    print(f"  Feature count after One-Hot: {len(final_features)}")

    search_space = []
    for _ in range(args.n_trials):
        search_space.append({
            'C': random.choice([0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]),
            'penalty': random.choice(['l1', 'l2']),
        })

    results = []
    for i, params in enumerate(search_space):
        result = run_trial(i, params, train_df, test_df, y_train_full, final_features, args.seed)
        results.append(result)
        print(f"  Trial {i + 1:3d}/{args.n_trials} | CV={result['CV']:.5f} | C={params['C']:<7} | Penalty={params['penalty']}")

    results.sort(key=lambda x: x['CV'], reverse=True)

    rank10_idx = min(9, len(results) - 1)
    rank50_idx = min(49, len(results) - 1)
    targets = [("Rank1", results[0]), ("Rank10", results[rank10_idx]), ("Rank50", results[rank50_idx])]
    out_dir = args.output_dir

    print("\n" + "=" * 90)
    print("  LogisticRegression Grid Search Results")
    print("=" * 90)
    for rank_name, r in targets:
        p = r['params']
        print(f"  {rank_name}: CV={r['CV']:.5f} | C={p['C']} | Penalty={p['penalty']}")
        file_name = os.path.join(out_dir,
            f"submission_lr_grid_{rank_name}_C{p['C']}_{p['penalty']}.csv")
        pd.DataFrame({
            'PassengerId': passenger_ids,
            'Transported': r['preds']
        }).to_csv(file_name, index=False)
        print(f"  >>> Exported: {file_name}")
    print("=" * 90)


if __name__ == "__main__":
    main()
