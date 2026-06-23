# %%
# ---------------------------------------------------------
# Import required libraries
# ---------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# %%
# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
data = pd.read_csv("../data/01_canada_per_capita_income.csv")

# Display first few rows
data.head()

# %%
# ---------------------------------------------------------
# Prepare training and testing data
# ---------------------------------------------------------
X = data[["year"]]  # 2D feature matrix
y = data["per capita income (US$)"]  # 1D target vector

x_train, x_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=10
)

print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))

# %%
# ---------------------------------------------------------
# Train Linear Regression Model
# ---------------------------------------------------------
reg = LinearRegression()
reg.fit(x_train, y_train)

print("Model Coefficient (slope):", reg.coef_[0])
print("Model Intercept:", reg.intercept_)
print("Model Accuracy (R² score):", reg.score(x_test, y_test))

# %%
# ---------------------------------------------------------
# Regression Line Plot
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))

# Scatter actual data
plt.scatter(data["year"], data["per capita income (US$)"], marker="+", color="red")

# Regression line
plt.plot(data["year"], reg.predict(data[["year"]]), color="blue")

plt.xlabel("Year")
plt.ylabel("Per Capita Income (US$)")
plt.title("Linear Regression Fit")
plt.show()

# %%
# ---------------------------------------------------------
# Residual Error Plot
# ---------------------------------------------------------
y_pred = reg.predict(X)
residuals = y - y_pred

plt.figure(figsize=(8, 5))
plt.scatter(data["year"], residuals, color="purple")
plt.axhline(y=0, color="black", linestyle="--")
plt.xlabel("Year")
plt.ylabel("Residual Error")
plt.title("Residual Plot")
plt.show()
