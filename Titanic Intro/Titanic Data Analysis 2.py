import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the data
train_data = pd.read_csv('/Users/heracles/Documents/Python Projects/Kaggle Projects/Titanic Intro/input/train.csv')
test_data = pd.read_csv('/Users/heracles/Documents/Python Projects/Kaggle Projects/Titanic Intro/input/test.csv')


# Feature Engineering Function
def process_data(df):
    df = df.copy()

    # 1. Fill missing Fare values with the median
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # 2. Extract Titles (Mr, Mrs, Miss, Master) from the Name column
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    # Group rare titles together and standardize common ones
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                                       'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # 3. Fill missing Ages based on the median age of their Title group
    df['Age'] = df['Age'].fillna(df.groupby('Title')['Age'].transform('median'))

    # 4. Combine SibSp and Parch into a single FamilySize feature
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # 5. Fill missing Embarked values with the most common port ('S')
    df['Embarked'] = df['Embarked'].fillna('S')

    return df


# Process both datasets identically
train_data = process_data(train_data)
test_data = process_data(test_data)

y = train_data["Survived"]

# We now include our new engineered features
features = ["Pclass", "Sex", "Age", "Fare", "FamilySize", "Title", "Embarked"]

# Convert categorical variables into dummy/indicator variables
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# Ensure both dataframes have the exact same columns after getting dummies
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# Train the model (increased max_depth slightly for the new features)
model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=1)
model.fit(X, y)

# Generate predictions
predictions = model.predict(X_test)

# Save submission
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('/Users/heracles/Documents/Python Projects/Kaggle Projects/Titanic Intro/submission_2_0.csv', index=False)
print("Your advanced submission was successfully saved!")
