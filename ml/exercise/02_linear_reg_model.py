# %%
# ---------------------------------------------------------
# Import required libraries
# ---------------------------------------------------------
import math
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from word2number import w2n
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# %%
# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
# Dataset contains:
# - experience (string number-words)
# - test_score(out of 10)
# - interview_score(out of 10)
# - salary($)  → target variable
data = pd.read_csv("../data/02_linear_reg_model_hiring_multi_feature.csv")

# Display first 20 rows
data.head(20)

# %%
# ---------------------------------------------------------
# Handle missing values in test_score
# ---------------------------------------------------------
median = math.floor(data["test_score(out of 10)"].median())
data["test_score(out of 10)"].fillna(median, inplace=True)

data.head(20)

# %%
# ---------------------------------------------------------
# Convert experience column (word → number)
# ---------------------------------------------------------
data["experience"] = data["experience"].fillna("zero")
data["experience"] = data["experience"].apply(lambda x: w2n.word_to_num(x))

data.head(20)

# %%
# ---------------------------------------------------------
# Prepare training and testing data
# ---------------------------------------------------------
X = data[["experience", "test_score(out of 10)", "interview_score(out of 10)"]]
y = data["salary($)"]

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=10)

print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))

# %%
# ---------------------------------------------------------
# Train Linear Regression Model
# ---------------------------------------------------------
# Linear Regression fits:
#   salary = m1*experience + m2*test_score + m3*interview_score + b
reg = LinearRegression()
reg.fit(x_train, y_train)

print("Model Coefficients:", reg.coef_)  # slopes for each feature
print("Model Intercept:", reg.intercept_)  # base salary
print("Predicted salaries:", reg.predict(x_test))
print("Model Accuracy (R² score):", reg.score(x_test, y_test))

# %%
# ---------------------------------------------------------
# 3D Scatter Plot of all features vs salary
# ---------------------------------------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

x = data["experience"]
y1 = data["test_score(out of 10)"]
z = data["interview_score(out of 10)"]
salary = data["salary($)"]

scatter = ax.scatter(x, y1, z, c=salary, cmap="viridis", s=60)

ax.set_xlabel("Experience (years)")
ax.set_ylabel("Test Score (out of 10)")
ax.set_zlabel("Interview Score (out of 10)")
ax.set_title("3D Feature Plot with Salary as Color")

cbar = plt.colorbar(scatter)
cbar.set_label("Salary ($)")

plt.show()

# %%
# ---------------------------------------------------------
# Regression Line Plot (Experience vs Salary)
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))

plt.scatter(data["experience"], data["salary($)"], color="red", marker="+")
plt.plot(data["experience"], reg.predict(X), color="blue")

plt.xlabel("Experience (years)")
plt.ylabel("Salary ($)")
plt.title("Regression Line: Experience vs Salary")
plt.show()

# %%
# ---------------------------------------------------------
# Residual Error Plot
# ---------------------------------------------------------
y_pred = reg.predict(X)
residuals = y - y_pred

plt.figure(figsize=(8, 5))
plt.scatter(data["experience"], residuals, color="purple")
plt.axhline(y=0, color="black", linestyle="--")

plt.xlabel("Experience (years)")
plt.ylabel("Residual Error")
plt.title("Residual Plot")
plt.show()

# %%
# ---------------------------------------------------------
# Predict salaries for given inputs
# ---------------------------------------------------------
salary_1 = reg.predict([[1, 9, 6]])
salary_2 = reg.predict([[12, 10, 10]])

print("Salary for 1 yr exp, 9 test, 6 interview:", salary_1[0])
print("Salary for 12 yr exp, 10 test, 10 interview:", salary_2[0])
