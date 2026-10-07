import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

# 1. Load the data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Optional Feature Engineering: If standard Kaggle columns exist, this adds a strong signal
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

# 3. Format categorical columns for LightGBM
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# 4. Set tuned hyperparameters for tabular data
params = {
    'n_estimators': 3000,
    'learning_rate': 0.015,
    'max_depth': 8,
    'num_leaves': 64,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'objective': 'binary',
    'metric': 'auc'
}

# 5. Initialize Stratified K-Fold
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Array to store the averaged test predictions
test_predictions = np.zeros(len(X_test))

print(f"Starting {n_splits}-Fold Cross-Validation...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_encoded)):
    # Split data for this fold
    X_train, y_train = X.iloc[train_idx], y_encoded[train_idx]
    X_val, y_val = X.iloc[val_idx], y_encoded[val_idx]

    model = lgb.LGBMClassifier(**params)

    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False)
        ]
    )

    # Predict on the test set and divide by the number of splits to average them out
    test_predictions += model.predict_proba(X_test)[:, 1] / n_splits
    print(f"Fold {fold + 1} complete. Optimal trees used: {model.best_iteration_}")

# 6. Format and save the final blended submission
submission = pd.DataFrame({
    'id': test_ids,
    'satisfaction': test_predictions
})

submission.to_csv('submission_cv.csv', index=False)
print("Saved optimized predictions to submission_cv.csv")
