import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

# 1. Load the data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. Separate features and target
target_col = 'satisfaction'

# Keep IDs for the final output
test_ids = test_df['id']

# Drop target and unneeded ID columns from training features
X_train = train_df.drop(columns=['id', target_col])
y_train = train_df[target_col]

# Drop ID from testing features
X_test = test_df.drop(columns=['id'])

# 3. Encode the target variable
# If the target is string ('neutral or dissatisfied', 'satisfied'), this converts it to 0 and 1
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

# 4. Prepare categorical features
# LightGBM handles categorical features natively if cast to the 'category' data type
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

for col in categorical_features:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# 5. Initialize and train the LightGBM classifier
model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)

print("Training model...")
model.fit(X_train, y_train_encoded)

# 6. Generate probability predictions for ROC AUC scoring
# predict_proba returns an array of shape (n_samples, n_classes).
# [:, 1] extracts the probability of the positive class (class '1').
print("Generating predictions...")
predictions = model.predict_proba(X_test)[:, 1]

# 7. Format and save the submission
submission = pd.DataFrame({
    'id': test_ids,
    'satisfaction': predictions
})

submission.to_csv('submission.csv', index=False)
print("Saved predictions to submission.csv")
