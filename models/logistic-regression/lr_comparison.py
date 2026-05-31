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

warnings.filterwarnings("ignore")


DEFAULT_DATA_DIR = "spaceship-titanic-dataset"
DEFAULT_OUTPUT_DIR = os.path.join("models", "logistic-regression", "outputs")


def preprocess(data_dir):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))

    df = pd.concat([train.assign(is_train=1), test.assign(is_train=0)], ignore_index=True)
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['Num'] = pd.to_numeric(df['Num'], errors='coerce')

    for col in ['HomePlanet', 'Side', 'Deck', 'Destination']:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        df[col] = df[col].fillna(df[col].median())
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

    dbscan = DBSCAN(eps=0.666, min_samples=2)
    df['Topology_Cluster'] = dbscan.fit_predict(spatial_scaled)

    log_cols = ['RoomService', 'ShoppingMall', 'Spa', 'TotalSpend', 'LuxurySpend', 'Spend_per_Age']
    for col in log_cols:
        df[col] = np.log1p(df[col])

    df['CryoSleep_x_TotalSpend'] = df['CryoSleep'] * df['TotalSpend']
    df['CryoSleep_x_LuxurySpend'] = df['CryoSleep'] * df['LuxurySpend']

    onehot_cols = ['Deck', 'HomePlanet', 'Side', 'Topology_Cluster']
    for col in onehot_cols:
        df[col] = df[col].astype(str)
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)

    base_numeric = [
        'CryoSleep', 'Cabin_Region', 'Age_Group', 'ShoppingMall', 'Spend_per_Age',
        'Max_Spa_by_Destination', 'LuxurySpend', 'Mean_Age_by_Deck',
        'RoomService', 'Spa', 'TotalSpend', 'Max_Spa_by_Deck',
        'Mean_Spend_by_HomePlanet', 'Mean_Spend_by_Deck',
        'CryoSleep_x_TotalSpend', 'CryoSleep_x_LuxurySpend',
    ]
    onehot_features = [c for c in df.columns if any(
        c.startswith(p + '_') for p in ['Deck', 'HomePlanet', 'Side', 'Topology_Cluster']
    )]
    final_features = base_numeric + onehot_features

    y_train_full = train['Transported'].astype(int).values

    train_df = df[df['is_train'] == 1].reset_index(drop=True)
    test_df = df[df['is_train'] == 0].reset_index(drop=True)

    return train_df, test_df, y_train_full, final_features, test['PassengerId']


def parse_args():
    parser = argparse.ArgumentParser(description="LR fixed-config comparison run.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="Directory containing train.csv and test.csv.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory for predictions and metrics.")
    parser.add_argument("--seed", type=int, default=78,
                        help="Random seed for reproducibility.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(">>> LogisticRegression comparison run")
    print(f"    data_dir   = {args.data_dir}")
    print(f"    output_dir = {args.output_dir}")
    print(f"    seed       = {args.seed}")

    train_df, test_df, y_train_full, final_features, passenger_ids = preprocess(args.data_dir)
    print(f"    feature_count = {len(final_features)}")

    lr_params = {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'lbfgs',
        'max_iter': 1000,
        'random_state': args.seed,
    }
    print(f"    lr_params  = {lr_params}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train_df))
    fold_accuracies = []
    scaler = StandardScaler()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df, y_train_full)):
        X_tr = train_df[final_features].iloc[train_idx].copy()
        X_va = train_df[final_features].iloc[val_idx].copy()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)
        model_cv = LogisticRegression(**lr_params)
        model_cv.fit(X_tr_scaled, y_train_full[train_idx])
        oof_preds[val_idx] = model_cv.predict(X_va_scaled)
        fold_acc = accuracy_score(y_train_full[val_idx], oof_preds[val_idx])
        fold_accuracies.append(fold_acc)
        print(f"    fold {fold_idx + 1}/5 acc = {fold_acc:.5f}")

    cv_acc = accuracy_score(y_train_full, oof_preds)

    X_train_all = scaler.fit_transform(train_df[final_features])
    X_test_all = scaler.transform(test_df[final_features])
    model_full = LogisticRegression(**lr_params)
    model_full.fit(X_train_all, y_train_full)
    preds = [bool(p) for p in model_full.predict(X_test_all)]

    pd.DataFrame({'PassengerId': passenger_ids, 'Transported': preds}).to_csv(
        os.path.join(args.output_dir, "submission_lr_comparison.csv"), index=False)
    pd.DataFrame({
        "metric": ["stratified_cv_acc"] + [f"fold{i+1}" for i in range(len(fold_accuracies))],
        "value":  [cv_acc] + fold_accuracies,
    }).to_csv(os.path.join(args.output_dir, "metrics_lr_comparison.csv"), index=False)

    print(f"    stratified_5fold_cv_acc = {cv_acc:.5f}")
    print(f">>> Saved submission to: {os.path.join(args.output_dir, 'submission_lr_comparison.csv')}")


if __name__ == "__main__":
    main()
