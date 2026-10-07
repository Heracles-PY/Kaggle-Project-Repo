import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
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

# LightGBM requires pandas 'category' type
X_lgb = X.copy()
X_test_lgb = X_test.copy()
for col in cat_cols:
    X_lgb[col] = X_lgb[col].astype('category')
    X_test_lgb[col] = X_test_lgb[col].astype('category')

# CatBoost prefers string types for categories
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
    'subsample': 0.8,
    'colsample_bytree': 0.8,
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

# 5. Initialize Stratified K-Fold
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Arrays to store the averaged test predictions
lgb_predictions = np.zeros(len(X_test))
cb_predictions = np.zeros(len(X_test))

print(f"Starting {n_splits}-Fold Cross-Validation Ensemble...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_encoded)):
    print(f"\n--- Training Fold {fold + 1} ---")

    # --- LightGBM Training ---
    X_train_lgb, y_train_lgb = X_lgb.iloc[train_idx], y_encoded[train_idx]
    X_val_lgb, y_val_lgb = X_lgb.iloc[val_idx], y_encoded[val_idx]

    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(
        X_train_lgb, y_train_lgb,
        eval_set=[(X_val_lgb, y_val_lgb)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )

    lgb_predictions += model_lgb.predict_proba(X_test_lgb)[:, 1] / n_splits
    print(f"LightGBM Fold {fold + 1} complete. Trees: {model_lgb.best_iteration_}")

    # --- CatBoost Training ---
    X_train_cb, y_train_cb = X_cb.iloc[train_idx], y_encoded[train_idx]
    X_val_cb, y_val_cb = X_cb.iloc[val_idx], y_encoded[val_idx]

    # Pools are CatBoost's highly optimized data structures
    train_pool = Pool(X_train_cb, y_train_cb, cat_features=cat_cols)
    val_pool = Pool(X_val_cb, y_val_cb, cat_features=cat_cols)
    test_pool = Pool(X_test_cb, cat_features=cat_cols)

    model_cb = CatBoostClassifier(**cb_params)
    model_cb.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=100
    )

    cb_predictions += model_cb.predict_proba(test_pool)[:, 1] / n_splits
    print(f"CatBoost Fold {fold + 1} complete. Trees: {model_cb.get_best_iteration()}")

# 6. Blend the predictions
print("\nBlending predictions...")
final_predictions = (lgb_predictions + cb_predictions) / 2

# 7. Format and save the final blended submission
submission = pd.DataFrame({
    'id': test_ids,
    'satisfaction': final_predictions
})

submission.to_csv('submission_ensemble.csv', index=False)
print("Saved optimized ensemble predictions to submission_ensemble.csv")
