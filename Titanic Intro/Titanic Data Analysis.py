# Data Analysis Predicting Who Would Survive On The Titanic
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import os
for dirname, _, filenames in os.walk('/Users/heracles/Documents/Python Projects/Kaggle Projects/Titanic Intro/input'):
    for filename in filenames:
        os.path.join(dirname, filename)

train_data = pd.read_csv('/Users/heracles/Documents/Python Projects/Kaggle Projects/Titanic Intro/input/train.csv')
train_data.head()

test_data = pd.read_csv('/Users/heracles/Documents/Python Projects/Kaggle Projects/Titanic Intro/input/test.csv')
test_data.head()

# Find the percentage of women that survived
# women = train_data.loc[train_data.Sex == 'female']["Survived"]
# rate_women = sum(women)/len(women)
# print("% of women who survived:", rate_women)
#
# Find the percentage of men that survived
# men = train_data.loc[train_data.Sex == 'male']["Survived"]
# rate_men = sum(men)/len(men)
# print("% of men who survived:", rate_men)

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('/Users/heracles/Documents/Python Projects/Kaggle Projects/Titanic Intro/submission.csv', index=False)
print("Your submission was successfully saved!")