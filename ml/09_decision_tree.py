# %%
# Generated from: 9_decision_tree.ipynb
# This script demonstrates how to use a Decision Tree Classifier to predict
# whether a person earns more than 100k based on company, job role, and degree.

# A Decision Tree is a machine‑learning algorithm that makes predictions by repeatedly splitting the data into smaller groups based on feature values — just like a flowchart.
# It works by asking a sequence of yes/no or true/false questions until it reaches a final decision.

# Learns rules from the data
# Splits the dataset based on the most important features
# Works well with categorical data (like company, job, degree)
# Creates branches that represent decisions
# Ends in leaf nodes that represent predictions (e.g., salary > 100k or not)

import pandas as pd

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
df = pd.read_csv("../data/09_decision_tree_salaries.csv")
df.head()
# Dataset contains:
#   company, job, degree, salary_more_then_100k (target)

# %%
# ---------------------------------------------------------
# Separate input features and target column
# ---------------------------------------------------------
inputs = df.drop("salary_more_then_100k", axis="columns")
target = df["salary_more_then_100k"]
target.head()
# inputs → categorical features
# target → binary label (1 = salary > 100k, 0 = otherwise)

# %%
# ---------------------------------------------------------
# Encode categorical features using LabelEncoder
# ---------------------------------------------------------
from sklearn.preprocessing import LabelEncoder

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

# Convert text categories into numeric labels
inputs["company_n"] = le_company.fit_transform(inputs["company"])
inputs["job_n"] = le_job.fit_transform(inputs["job"])
inputs["degree_n"] = le_degree.fit_transform(inputs["degree"])

inputs.head()
# Now each categorical column has a corresponding numeric version

# %%
# ---------------------------------------------------------
# Prepare final feature set for training
# ---------------------------------------------------------
inputs_n = inputs.drop(["company", "job", "degree"], axis="columns")
inputs_n.head()
# Only numeric columns remain: company_n, job_n, degree_n

# %%
# ---------------------------------------------------------
# Train Decision Tree Classifier
# ---------------------------------------------------------
from sklearn import tree

model = tree.DecisionTreeClassifier()
model.fit(inputs_n, target)
# The model learns patterns linking company, job, degree → salary bracket


# ---------------------------------------------------------
# Evaluate model accuracy
# ---------------------------------------------------------
print("Evaluate model accuracy", model.score(inputs_n, target))
# Measures how well the model fits the training data


# ---------------------------------------------------------
# Make predictions
# ---------------------------------------------------------
# Predict salary > 100k for:
# Google (2), Computer Engineer (1), Bachelors (0)
print(
    "Predict salary > 100k for: Google (2), Computer Engineer (1), Bachelors (0)",
    model.predict([[2, 1, 0]]),
)

# Predict salary > 100k for:
# Google (2), Computer Engineer (1), Masters (1)
print(
    "Predict salary > 100k for: Google (2), Computer Engineer (1), Masters (1)",
    model.predict([[2, 1, 1]]),
)


# ---------------------------------------------------------
# Exercise
# ---------------------------------------------------------
# Build a decision tree model using the Titanic dataset to predict survival.
# Use the following columns:
#   - Pclass
#   - Sex
#   - Age
#   - Fare
#
# Your tasks:
#   1. Load the dataset
#   2. Encode categorical columns (e.g., Sex)
#   3. Train a DecisionTreeClassifier
#   4. Calculate model accuracy
#
# Dataset link:
# https://github.com/codebasics/py/blob/master/ML/9_decision_tree/Exercise/titanic.csv
#
# Image reference:
# <img src="titanic.jpg" height=200 width=400/>
