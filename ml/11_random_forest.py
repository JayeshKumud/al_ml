# %%
# Generated from: 11_random_forest.ipynb
# This script demonstrates how Random Forest Classifier can be used
# for multiclass classification using the sklearn digits dataset.
#
# RANDOM FOREST MODEL:
# --------------------
# Random Forest is an ensemble learning algorithm that builds multiple
# decision trees and combines their outputs. Each tree learns different
# patterns from random subsets of data and features.
#
# Key advantages:
#   - Handles high‑dimensional data well
#   - Reduces overfitting compared to a single decision tree
#   - Works well for both classification and regression
#   - Robust and accurate due to ensemble voting

import pandas as pd
from sklearn.datasets import load_digits

# ---------------------------------------------------------
# Load digits dataset (8x8 pixel handwritten digits)
# ---------------------------------------------------------
digits = load_digits()

# %%
# Inspect dataset structure
dir(digits)

# %%
# Visualize first few digit images
%matplotlib inline
import matplotlib.pyplot as plt

plt.gray()
for i in range(4):
    plt.matshow(digits.images[i])

# %%
# Create DataFrame from digit pixel values
df = pd.DataFrame(digits.data)
df.head()

# %%
# Add target column (digit labels 0–9)
df["target"] = digits.target
df.head()

# %%
# Display first 12 rows
df[0:12]

# ---------------------------------------------------------
# Prepare data for training
# ---------------------------------------------------------
X = df.drop("target", axis="columns")   # Features: 64 pixel values
y = df.target                           # Target: digit label

from sklearn.model_selection import train_test_split

# Split into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ---------------------------------------------------------
# Train Random Forest Classifier
# ---------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier

# n_estimators = number of trees in the forest
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train, y_train)

# %%
# Display model accuracy
print("Model accuracy:", model.score(X_test, y_test))

# %%
# Predict labels for test set
y_predicted = model.predict(X_test)
print("Sample predictions:", y_predicted[:10])

# ---------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_predicted)

# %%
# Display confusion matrix values
cm

# %%
# Visualize confusion matrix as heatmap
import seaborn as sn
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Truth")

# ---------------------------------------------------------
# Exercise
# ---------------------------------------------------------
# Use the iris dataset to train a Random Forest classifier.
#
# Tasks:
#   1. Train using default n_estimators=10 and measure accuracy.
#   2. Tune the number of trees (n_estimators) to find the best score.
#
# Dataset: from sklearn.datasets import load_iris
