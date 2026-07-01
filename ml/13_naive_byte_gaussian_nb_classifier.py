# %%
# ------------------------------------------------------------
# MODEL & ALGORITHM DOCUMENTATION — NAIVE BAYES (GAUSSIANNB)
# ------------------------------------------------------------
# MODEL NAME:
#   Naive Bayes Classifier (GaussianNB)
#
# MODEL DEFINITION:
#   Naive Bayes is a probabilistic machine learning algorithm
#   based on Bayes' Theorem. It assumes that all features are
#   independent (Naive assumption) and calculates the probability
#   of each class given the input features.
#
# MODEL FUNCTIONING (HOW IT WORKS INTERNALLY):
#
#   1. Calculates prior probability of each class
#        Example: Probability of survival vs non-survival.
#
#   2. Calculates likelihood:
#        P(feature value | class)
#        GaussianNB assumes features follow a normal distribution.
#
#   3. Computes posterior probability:
#        P(Class | Features)
#
#   4. Predicts the class with highest posterior probability.
#
# WHY NAIVE BAYES IS USED HERE:
#   • Works well with categorical + numeric data
#   • Fast and efficient
#   • Handles missing values well after imputation
#   • Suitable for Titanic dataset (simple features)
#

import pandas as pd

# %%
# ------------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------------
df = pd.read_csv("./data/14_naive_bayes_titanic_survival_prediction")
df.head()

# %%
# ------------------------------------------------------------
# DROP IRRELEVANT COLUMNS
# ------------------------------------------------------------
df.drop(
    [
        "PassengerId",
        "Name",
        "SibSp",
        "Parch",
        "Ticket",
        "Cabin",
        "Embarked",
    ],
    axis="columns",
    inplace=True,
)
df.head()

# %%
# ------------------------------------------------------------
# SEPARATE INPUTS AND TARGET
# ------------------------------------------------------------
inputs = df.drop("Survived", axis="columns")
target = df.Survived

# %%
# ------------------------------------------------------------
# CREATE DUMMY VARIABLES FOR SEX
# ------------------------------------------------------------
dummies = pd.get_dummies(inputs.Sex)
dummies.head(3)

# Add dummy columns to inputs
inputs = pd.concat([inputs, dummies], axis="columns")
inputs.head(3)

# Drop original Sex column + one dummy column (dummy variable trap)
inputs.drop(["Sex", "male"], axis="columns", inplace=True)
inputs.head(3)

# %%
# ------------------------------------------------------------
# CHECK FOR MISSING VALUES
# ------------------------------------------------------------
inputs.columns[inputs.isna().any()]

# %%
# ------------------------------------------------------------
# HANDLE MISSING AGE VALUES
# ------------------------------------------------------------
inputs.Age[:10]
inputs.Age = inputs.Age.fillna(inputs.Age.mean())
inputs.head()

# %%
# ------------------------------------------------------------
# TRAIN-TEST SPLIT
# ------------------------------------------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.3)

# %%
# ------------------------------------------------------------
# TRAIN NAIVE BAYES MODEL
# ------------------------------------------------------------
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)

# %%
# ------------------------------------------------------------
# MODEL ACCURACY
# ------------------------------------------------------------
print("Model Accuracy:", model.score(X_test, y_test))

# %%
# ------------------------------------------------------------
# SAMPLE TEST DATA (FIRST 10 ROWS)
# ------------------------------------------------------------
X_test[0:10]
y_test[0:10]

# %%
# ------------------------------------------------------------
# PREDICTIONS FOR FIRST 10 TEST ROWS
# ------------------------------------------------------------
print("Predicted Survival:", model.predict(X_test[0:10]))

# %%
# ------------------------------------------------------------
# PREDICTION PROBABILITIES
# ------------------------------------------------------------
print("Prediction Probabilities:")
model.predict_proba(X_test[:10])

# %%
# ------------------------------------------------------------
# CROSS-VALIDATION SCORE
# ------------------------------------------------------------
from sklearn.model_selection import cross_val_score

print("Cross Validation Scores:")
cross_val_score(GaussianNB(), X_train, y_train, cv=5)
