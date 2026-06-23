# %%
# Generated from: 12_k_fold.ipynb
# This script demonstrates how to evaluate ML models using
# K-Fold and Stratified K-Fold cross validation.
#
# WHY K-FOLD?
# -----------
# Instead of training/testing once, K-Fold splits the dataset into K parts
# and trains the model K times. This gives a more reliable performance score.
#
# MODELS USED:
# 1. Logistic Regression  - Linear classifier, fast, interpretable
# 2. SVM (Support Vector Machine) - Margin-based classifier, strong for complex data
# 3. Random Forest - Ensemble of decision trees, robust and accurate

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Load digits dataset (8x8 pixel handwritten digits)
# ---------------------------------------------------------
digits = load_digits()

# %%
# Display dataset structure
print("Feature shape:", digits.data.shape)
print("Target shape:", digits.target.shape)

# ---------------------------------------------------------
# Train-test split (simple evaluation)
# ---------------------------------------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.3
)

# ---------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------
lr = LogisticRegression(solver="liblinear", multi_class="ovr")
lr.fit(X_train, y_train)

# %%
print("Logistic Regression Score:", lr.score(X_test, y_test))

# ---------------------------------------------------------
# SVM
# ---------------------------------------------------------
svm = SVC(gamma="auto")
svm.fit(X_train, y_train)

# %%
print("SVM Score:", svm.score(X_test, y_test))

# ---------------------------------------------------------
# Random Forest
# ---------------------------------------------------------
rf = RandomForestClassifier(n_estimators=40)
rf.fit(X_train, y_train)

# %%
print("Random Forest Score:", rf.score(X_test, y_test))

# ---------------------------------------------------------
# K-Fold Cross Validation (Basic Example)
# ---------------------------------------------------------
from sklearn.model_selection import KFold

kf = KFold(n_splits=3)

# %%
print("Basic KFold Example:")
for train_index, test_index in kf.split([1, 2, 3, 4, 5, 6, 7, 8, 9]):
    print("Train:", train_index, " Test:", test_index)

# ---------------------------------------------------------
# Stratified K-Fold for digits dataset
# ---------------------------------------------------------
# StratifiedKFold ensures each fold has equal class distribution
from sklearn.model_selection import StratifiedKFold

folds = StratifiedKFold(n_splits=3)


# Helper function to compute score
def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


scores_logistic = []
scores_svm = []
scores_rf = []

# Perform 3-fold CV
for train_index, test_index in folds.split(digits.data, digits.target):
    X_train, X_test = digits.data[train_index], digits.data[test_index]
    y_train, y_test = digits.target[train_index], digits.target[test_index]

    scores_logistic.append(
        get_score(
            LogisticRegression(solver="liblinear", multi_class="ovr"),
            X_train,
            X_test,
            y_train,
            y_test,
        )
    )
    scores_svm.append(get_score(SVC(gamma="auto"), X_train, X_test, y_train, y_test))
    scores_rf.append(
        get_score(
            RandomForestClassifier(n_estimators=40), X_train, X_test, y_train, y_test
        )
    )

# %%
print("Logistic Regression KFold Scores:", scores_logistic)

# %%
print("SVM KFold Scores:", scores_svm)

# %%
print("Random Forest KFold Scores:", scores_rf)

# ---------------------------------------------------------
# cross_val_score (simplified K-Fold)
# ---------------------------------------------------------
from sklearn.model_selection import cross_val_score

# %%
print("cross_val_score - Logistic Regression:")
print(
    cross_val_score(
        LogisticRegression(solver="liblinear", multi_class="ovr"),
        digits.data,
        digits.target,
        cv=3,
    )
)

# %%
print("cross_val_score - SVM:")
print(cross_val_score(SVC(gamma="auto"), digits.data, digits.target, cv=3))

# %%
print("cross_val_score - Random Forest:")
print(
    cross_val_score(
        RandomForestClassifier(n_estimators=40), digits.data, digits.target, cv=3
    )
)

# ---------------------------------------------------------
# Parameter tuning using K-Fold
# ---------------------------------------------------------
scores1 = cross_val_score(
    RandomForestClassifier(n_estimators=5), digits.data, digits.target, cv=10
)
scores2 = cross_val_score(
    RandomForestClassifier(n_estimators=20), digits.data, digits.target, cv=10
)
scores3 = cross_val_score(
    RandomForestClassifier(n_estimators=30), digits.data, digits.target, cv=10
)
scores4 = cross_val_score(
    RandomForestClassifier(n_estimators=40), digits.data, digits.target, cv=10
)

# %%
print("Avg Score (5 trees):", np.average(scores1))
print("Avg Score (20 trees):", np.average(scores2))
print("Avg Score (30 trees):", np.average(scores3))
print("Avg Score (40 trees):", np.average(scores4))

# Best result typically around 40 trees

# ---------------------------------------------------------
# Exercise
# ---------------------------------------------------------
# Use iris dataset and evaluate:
# 1. Logistic Regression
# 2. SVM
# 3. Decision Tree
# 4. Random Forest
# using cross_val_score and compare performance.
