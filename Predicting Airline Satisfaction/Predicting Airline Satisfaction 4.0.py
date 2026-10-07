import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

# 1. Load the data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Feature Engineering
if 'Departure Delay in Minutes' in train_df.columns and 'Arrival Delay in Minutes' in train_df.columns:
    for df in [train_df, test_df]:
        df['Total_Delay'] = df['Departure Delay in Minutes'] + df['Arrival Delay in Minutes'].fillna(0)

target_col = 'satisfaction'
test_ids = test_df['id']

# 2. Separate features and target
X = train_df.drop(columns=['id', target_col])
y = train_df[target_col]
X_test = test_df.drop(columns=['id'])

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 3. Identify and format categorical columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# LightGBM and XGBoost require pandas 'category' type
X_tree = X.copy()
X_test_tree = X_test.copy()
for col in cat_cols:
    X_tree[col] = X_tree[col].astype('category')
    X_test_tree[col] = X_test_tree[col].astype('category')

# CatBoost prefers string types
X_cb = X.copy()
X_test_cb = X_test.copy()
for col in cat_cols:
    X_cb[col] = X_cb[col].astype(str)
    X_test_cb[col] = X_test_cb[col].astype(str)

# 4. Set tuned hyperparameters
lgb_params = {
    'n_estimators': 3000,
    'learning_rate': 0.015,
    'max_depth': 8,
    'num_leaves': 64,
    'random_state': 42,
    'n_jobs': -1,
    'objective': 'binary',
    'metric': 'auc',
    'verbose': -1
}

cb_params = {
    'iterations': 3000,
    'learning_rate': 0.015,
    'depth': 8,
    'eval_metric': 'AUC',
    'random_seed': 42,
    'task_type': 'CPU',
    'verbose': False
}

xgb_params = {
    'n_estimators': 3000,
    'learning_rate': 0.015,
    'max_depth': 8,
    'eval_metric': 'auc',
    'random_state': 42,
    'tree_method': 'hist',
    'enable_categorical': True,
    'early_stopping_rounds': 100
}

# 5. Initialize Stratified K-Fold
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Arrays to store the averaged test predictions
lgb_predictions = np.zeros(len(X_test))
cb_predictions = np.zeros(len(X_test))
xgb_predictions = np.zeros(len(X_test))

print(f"Starting {n_splits}-Fold Cross-Validation Trifecta...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_encoded)):
    print(f"\n--- Training Fold {fold + 1} ---")

    # --- LightGBM ---
    X_train_tree, y_train = X_tree.iloc[train_idx], y_encoded[train_idx]
    X_val_tree, y_val = X_tree.iloc[val_idx], y_encoded[val_idx]

    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(
        X_train_tree, y_train,
        eval_set=[(X_val_tree, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )
    lgb_predictions += model_lgb.predict_proba(X_test_tree)[:, 1] / n_splits
    print(f"LightGBM Fold {fold + 1} complete.")

    # --- XGBoost ---
    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(
        X_train_tree, y_train,
        eval_set=[(X_val_tree, y_val)],
        verbose=False
    )
    xgb_predictions += model_xgb.predict_proba(X_test_tree)[:, 1] / n_splits
    print(f"XGBoost Fold {fold + 1} complete.")

    # --- CatBoost ---
    X_train_cb, X_val_cb = X_cb.iloc[train_idx], X_cb.iloc[val_idx]

    train_pool = Pool(X_train_cb, y_train, cat_features=cat_cols)
    val_pool = Pool(X_val_cb, y_val, cat_features=cat_cols)
    test_pool = Pool(X_test_cb, cat_features=cat_cols)

    model_cb = CatBoostClassifier(**cb_params)
    model_cb.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=100
    )
    cb_predictions += model_cb.predict_proba(test_pool)[:, 1] / n_splits
    print(f"CatBoost Fold {fold + 1} complete.")

# 6. Blend the predictions equally
print("\nBlending predictions (LightGBM + CatBoost + XGBoost)...")
final_predictions = (lgb_predictions + cb_predictions + xgb_predictions) / 3

# 7. Format and save
submission = pd.DataFrame({
    'id': test_ids,
    'satisfaction': final_predictions
})

submission.to_csv('submission_trifecta.csv', index=False)
print("Saved optimized trifecta predictions to submission_trifecta.csv")
