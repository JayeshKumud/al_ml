# %%
# ------------------------------------------------------------
# MODEL & ALGORITHM DOCUMENTATION — L1 & L2 REGULARIZATION
# ------------------------------------------------------------
# WHAT IS REGULARIZATION?
#   Regularization is a technique used to reduce overfitting in
#   machine learning models by adding a penalty term to the loss
#   function. It prevents the model from learning noise and helps
#   generalize better on unseen data.
#
# WHY REGULARIZATION IS NEEDED:
#   • Linear Regression often overfits when dataset has many
#     features or noisy data.
#   • Overfitting → high training score but low test score.
#
# TYPES OF REGULARIZATION:
#
#   L1 Regularization (Lasso Regression)
#     • Adds penalty = α * |weights|
#     • Shrinks some weights to zero → feature selection
#
#   L2 Regularization (Ridge Regression)
#     • Adds penalty = α * (weights²)
#     • Shrinks weights smoothly → reduces model complexity
# ------------------------------------------------------------


# %%
# ------------------------------------------------------------
# IMPORT LIBRARIES
# ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

# %%
# ------------------------------------------------------------
# LOAD MELBOURNE HOUSING DATASET
# ------------------------------------------------------------
dataset = pd.read_csv("./data/16_regularization_melbourne_housing_full.csv")
dataset.head()  # Important DataFrame

# %%
# ------------------------------------------------------------
# CHECK UNIQUE VALUES
# ------------------------------------------------------------
dataset.nunique()

# %%
# ------------------------------------------------------------
# SELECT RELEVANT COLUMNS
# ------------------------------------------------------------
cols_to_use = [
    "Suburb",
    "Rooms",
    "Type",
    "Method",
    "SellerG",
    "Regionname",
    "Propertycount",
    "Distance",
    "CouncilArea",
    "Bedroom2",
    "Bathroom",
    "Car",
    "Landsize",
    "BuildingArea",
    "Price",
]

dataset = dataset[cols_to_use]
dataset.head()  # Important DataFrame

# %%
# ------------------------------------------------------------
# CHECK SHAPE
# ------------------------------------------------------------
dataset.shape

# %%
# ------------------------------------------------------------
# CHECK MISSING VALUES
# ------------------------------------------------------------
dataset.isna().sum()

# %%
# ------------------------------------------------------------
# HANDLE MISSING VALUES
# ------------------------------------------------------------
cols_to_fill_zero = ["Propertycount", "Distance", "Bedroom2", "Bathroom", "Car"]
dataset[cols_to_fill_zero] = dataset[cols_to_fill_zero].fillna(0)

dataset["Landsize"] = dataset["Landsize"].fillna(dataset.Landsize.mean())
dataset["BuildingArea"] = dataset["BuildingArea"].fillna(dataset.BuildingArea.mean())

# Drop rows where Price is missing
dataset.dropna(inplace=True)

dataset.shape

# %%
# ------------------------------------------------------------
# ONE-HOT ENCODE CATEGORICAL FEATURES
# ------------------------------------------------------------
dataset = pd.get_dummies(dataset, drop_first=True)
dataset.head()  # Important DataFrame

# %%
# ------------------------------------------------------------
# SPLIT INTO TRAIN & TEST SETS
# ------------------------------------------------------------
X = dataset.drop("Price", axis=1)
y = dataset["Price"]

from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=2)

# %%
# ------------------------------------------------------------
# LINEAR REGRESSION (BASELINE MODEL)
# ------------------------------------------------------------
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(train_X, train_y)

print("Linear Regression Test Score:", reg.score(test_X, test_y))
print("Linear Regression Train Score:", reg.score(train_X, train_y))

# Observation:
# Training score = high
# Test score = very low → model is overfitting

# %%
# ------------------------------------------------------------
# LASSO REGRESSION (L1 REGULARIZATION)
# ------------------------------------------------------------
from sklearn import linear_model

lasso_reg = linear_model.Lasso(alpha=50, max_iter=100, tol=0.1)
lasso_reg.fit(train_X, train_y)

print("Lasso Regression Test Score:", lasso_reg.score(test_X, test_y))
print("Lasso Regression Train Score:", lasso_reg.score(train_X, train_y))

# %%
# ------------------------------------------------------------
# RIDGE REGRESSION (L2 REGULARIZATION)
# ------------------------------------------------------------
from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=50, max_iter=100, tol=0.1)
ridge_reg.fit(train_X, train_y)

print("Ridge Regression Test Score:", ridge_reg.score(test_X, test_y))
print("Ridge Regression Train Score:", ridge_reg.score(train_X, train_y))

# %%
# ------------------------------------------------------------
# FINAL CONCLUSION
# ------------------------------------------------------------
print("""
Conclusion:
- Linear Regression overfits heavily.
- Lasso (L1) reduces overfitting by shrinking some coefficients to zero.
- Ridge (L2) reduces overfitting by shrinking coefficients smoothly.
- Regularization improves generalization and is widely used in ML & Neural Networks.
""")
