# %%
# ------------------------------------------------------------
# HR Analytics – Logistic Regression Model
# Predict whether an employee will leave the company
# Dataset: https://www.kaggle.com/giripujar/hr-analytics
# ------------------------------------------------------------
# This script performs:
# 1. Data loading & cleaning
# 2. Feature selection & encoding
# 3. Train-test split
# 4. Logistic Regression model training
# 5. Model evaluation (accuracy + predictions)
# ------------------------------------------------------------

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from IPython.display import display

# %%
# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------
df = pd.read_csv("../data/07_logistic_reg_model_HR_comma_sep.csv")

# Display first few rows to understand structure
display(df.head())

# %%
# ------------------------------------------------------------
# 2. Drop irrelevant or weakly correlated columns
# ------------------------------------------------------------
# These columns were removed because:
# - last_evaluation, number_project, average_montly_hours, time_spend_company
#   do not strongly differentiate between employees who left vs stayed.
# - Work_accident has minimal predictive power.
df_clean = df.drop(
    [
        "last_evaluation",
        "number_project",
        "average_montly_hours",
        "time_spend_company",
        "Work_accident",
    ],
    axis="columns",
)

display(df_clean.head())

# %%
# ------------------------------------------------------------
# 3. Encode categorical variables (Department, salary)
# ------------------------------------------------------------
# Logistic Regression requires numeric inputs.
# LabelEncoder converts text → numeric labels.
dep_encoder = LabelEncoder()
sal_encoder = LabelEncoder()

df_clean["Department"] = dep_encoder.fit_transform(df_clean["Department"])
df_clean["salary"] = sal_encoder.fit_transform(df_clean["salary"])

display(df_clean)

# %%
# ------------------------------------------------------------
# 4. Select Features (X) and Target (y)
# ------------------------------------------------------------
# Based on EDA, the following features were selected:
# - satisfaction_level: strong negative correlation with attrition
# - promotion_last_5years: promotions reduce attrition
# - Department: encoded categorical feature
# - salary: encoded categorical feature
#
# Target:
# - left (0 = stayed, 1 = left)
x = df_clean[["satisfaction_level", "promotion_last_5years", "Department", "salary"]]
y = df_clean["left"]

display("Feature Matrix (X)", x.head())
display("Target Vector (y)", y.head())

# %%
# ------------------------------------------------------------
# 5. Train-Test Split
# ------------------------------------------------------------
# 80% training, 20% testing
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=10
)

print("Training samples:", x_train.shape)
print("Testing samples:", x_test.shape)
print("Training labels:", y_train.shape)
print("Testing labels:", y_test.shape)

# %%
# ------------------------------------------------------------
# 6. Logistic Regression Model
# ------------------------------------------------------------
# Logistic Regression is a classification algorithm used when the target
# variable is binary (0 or 1). It predicts the probability of an employee
# leaving the company based on the selected features.
#
# The model uses the sigmoid function to convert linear output → probability.
model = LogisticRegression(max_iter=200)
model.fit(x_train, y_train)

# %%
# ------------------------------------------------------------
# 7. Predictions on Test Data
# ------------------------------------------------------------
predicted = model.predict(x_test)
print("Predicted values (first 20):", predicted[:20])

# %%
# ------------------------------------------------------------
# 8. Model Accuracy
# ------------------------------------------------------------
accuracy = model.score(x_test, y_test)
print("Model Accuracy:", accuracy)
