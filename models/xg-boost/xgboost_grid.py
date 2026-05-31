import argparse
import os
import random
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')


DEFAULT_DATA_DIR = "spaceship-titanic-dataset"
DEFAULT_OUTPUT_DIR = os.path.join("models", "xg-boost", "outputs")


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


def run_trial(args):
    trial_id, params, train_df, test_df, y_train_full, final_features, seed = args

    xgb_params = {
        'n_estimators': 1358,
        'learning_rate': 0.0666,
        'max_depth': params['max_depth'],
        'reg_lambda': params['reg_lambda'],
        'subsample': params['subsample'],
        'max_bin': params['max_bin'],
        'colsample_bytree': params['colsample_bytree'],
        'min_child_weight': params['min_child_weight'],
        'reg_alpha': params['reg_alpha'],
        'tree_method': 'hist',
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_jobs': 1,
        'verbosity': 0,
        'random_state': seed,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train_df))
    for train_idx, val_idx in skf.split(train_df, y_train_full):
        X_tr = train_df[final_features].iloc[train_idx]
        X_va = train_df[final_features].iloc[val_idx]
        model_cv = XGBClassifier(**xgb_params)
        model_cv.fit(X_tr, y_train_full[train_idx])
        oof_preds[val_idx] = model_cv.predict(X_va)

    cv_acc = accuracy_score(y_train_full, oof_preds)

    model_full = XGBClassifier(**xgb_params)
    model_full.fit(train_df[final_features], y_train_full)
    preds = [bool(p) for p in model_full.predict(test_df[final_features])]

    return {'Trial': trial_id, 'CV': cv_acc, 'params': params, 'preds': preds}


def parse_args():
    parser = argparse.ArgumentParser(description="XGBoost randomized grid search.")
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

    print(">>> XGBoost Randomized Grid Search")
    print(f"    data_dir   = {args.data_dir}")
    print(f"    output_dir = {args.output_dir}")
    print(f"    seed       = {args.seed}")
    print(f"    n_trials   = {args.n_trials}")
    print(">>> Fixed: n_estimators=1358, learning_rate=0.0666, eps=0.666")

    train_df, test_df, y_train_full, final_features, passenger_ids = preprocess(args.data_dir)

    search_space = []
    for _ in range(args.n_trials):
        search_space.append({
            'max_depth': random.randint(3, 8),
            'reg_lambda': round(random.uniform(1.0, 15.0), 3),
            'subsample': round(random.uniform(0.6, 1.0), 3),
            'max_bin': random.randint(100, 512),
            'colsample_bytree': round(random.uniform(0.6, 1.0), 3),
            'min_child_weight': random.randint(1, 10),
            'reg_alpha': round(random.uniform(0.0, 5.0), 3),
        })

    results = []
    for i, params in enumerate(search_space):
        result = run_trial((i, params, train_df, test_df, y_train_full, final_features, args.seed))
        results.append(result)
        print(f"  Trial {i + 1:3d}/{args.n_trials} | CV={result['CV']:.5f} | Depth={params['max_depth']} "
              f"| Lambda={params['reg_lambda']:.3f} | Sub={params['subsample']:.3f}")

    results.sort(key=lambda x: x['CV'], reverse=True)

    rank10_idx = min(9, len(results) - 1)
    rank50_idx = min(49, len(results) - 1)
    targets = [("Rank1", results[0]), ("Rank10", results[rank10_idx]), ("Rank50", results[rank50_idx])]
    out_dir = args.output_dir

    print("\n" + "=" * 120)
    print("  XGBoost Grid Search Results")
    print("=" * 120)
    for rank_name, r in targets:
        p = r['params']
        print(f"  {rank_name}: CV={r['CV']:.5f} | Depth={p['max_depth']} | Lambda={p['reg_lambda']} | "
              f"Sub={p['subsample']} | Bin={p['max_bin']} | ColSample={p['colsample_bytree']} | "
              f"MinChild={p['min_child_weight']} | Alpha={p['reg_alpha']}")
        file_name = os.path.join(out_dir,
            f"submission_xgb_grid_{rank_name}_d{p['max_depth']}_lam{p['reg_lambda']}"
            f"_sub{p['subsample']}_mcw{p['min_child_weight']}.csv")
        pd.DataFrame({
            'PassengerId': passenger_ids,
            'Transported': r['preds']
        }).to_csv(file_name, index=False)
        print(f"  >>> Exported: {file_name}")
    print("=" * 120)


if __name__ == "__main__":
    main()
