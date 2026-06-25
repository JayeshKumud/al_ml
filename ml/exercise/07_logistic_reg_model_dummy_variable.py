# %%
# ------------------------------------------------------------
# HR Analytics – Logistic Regression Model
# Predict whether an employee will leave the company
# Dataset: https://www.kaggle.com/giripujar/hr-analytics
# ------------------------------------------------------------

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from IPython.display import display

# %%
# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------
df = pd.read_csv("../data/07_logistic_reg_model_HR_comma_sep.csv")

# Display first few rows to understand structure
df.head()

# %%
# ------------------------------------------------------------
# 2. Data Exploration
# ------------------------------------------------------------

# Employees who left
left = df[df.left == 1]
print("Employees who left:", left.shape)

# Employees who stayed
retained = df[df.left == 0]
print("Employees retained:", retained.shape)

# Average values grouped by 'left'
grouped_table = df.groupby("left").mean(numeric_only=True)
display(grouped_table)

# From the grouped mean table, we can clearly observe the following:
#
# 1. Satisfaction Level:
#    - Employees who LEFT the company have a much lower average satisfaction level (~0.44)
#    - Employees who STAYED have a higher satisfaction level (~0.66)
#    - This indicates that satisfaction level is a strong predictor of attrition.
#
# 2. Average Monthly Hours:
#    - Employees who LEFT work more hours on average (~207)
#    - Employees who STAYED work fewer hours (~199)
#    - Overworking may contribute to burnout, increasing the likelihood of leaving.
#
# 3. Promotion in Last 5 Years:
#    - Employees who received promotions are more likely to stay.
#    - The average promotion rate is higher for retained employees.
#    - This suggests that career growth reduces attrition.
#
# These three variables show clear separation between employees who left vs stayed,
# making them strong candidates for our predictive model.


# %%
# ------------------------------------------------------------
# 3. Impact of Salary on Employee Retention
# ------------------------------------------------------------
pd.crosstab(df.salary, df.left).plot(kind="bar")
plt.title("Salary vs Employee Attrition")
plt.xlabel("Salary Level")
plt.ylabel("Count")
plt.show()

# %%
# ------------------------------------------------------------
# 4. Impact of Department on Employee Retention
# ------------------------------------------------------------
pd.crosstab(df.Department, df.left).plot(kind="bar")
plt.title("Department vs Employee Attrition")
plt.xlabel("Department")
plt.ylabel("Count")
plt.show()

# %%
# ------------------------------------------------------------
# 5. Select Features for the Model
# ------------------------------------------------------------
subdf = df[
    ["satisfaction_level", "average_montly_hours", "promotion_last_5years", "salary"]
]

subdf.head()

# %%
# ------------------------------------------------------------
# 6. Convert Salary (Categorical) into Dummy Variables
# ------------------------------------------------------------
salary_dummies = pd.get_dummies(subdf.salary, prefix="salary", dtype=int)

df_with_dummies = pd.concat([subdf, salary_dummies], axis="columns")

df_with_dummies.drop("salary", axis="columns", inplace=True)

df_with_dummies.head()

# %%
# ------------------------------------------------------------
# 7. Prepare Feature Matrix (X) and Target Vector (y)
# ------------------------------------------------------------
X = df_with_dummies
y = df.left

X.head()

# %%
# ------------------------------------------------------------
# 8. Train-Test Split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.3, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# %%
# ------------------------------------------------------------
# 9. Logistic Regression Model
# ------------------------------------------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

predicted = model.predict(X_test)
print("Predicted values (first 20):", predicted[:20])

# %%
# ------------------------------------------------------------
# 10. Model Accuracy
# ------------------------------------------------------------
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)
