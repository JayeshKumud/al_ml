# %%
# Linear regression is one of the simplest and most important concepts in machine learning.
# At its core, it tries to draw a straight line through data points so you can predict a value based on an input.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Load the dataset
df = pd.read_csv("./data/homeprices.csv")

# Plot the raw data
plt.xlabel("Area in Sq ft")
plt.ylabel("Price")
plt.scatter(df.area, df.price, color="red", marker="+")
# Creates a scatter plot to visualize the relationship between area and price.

# Create the Linear Regression model
reg = linear_model.LinearRegression()
# Initializes a linear regression model object.

# Train (fit) the model
reg.fit(df[["area"]], df.price)
# df[["area"]] → 2D input feature matrix
# df.price → target values
# The model learns the best‑fit line using the formula: y = mx + b

# 𝑚 = slope (model coefficient) → reg.coef_
# 𝑏 = intercept → reg.intercept_
# 𝑥 = area
# 𝑦 = predicted price

print("model coefficient", reg.coef_)
print("intercept", reg.intercept_)

# After training, the model has:
# reg.coef_ → value of m
# reg.intercept_ → value of b

reg.predict([[3300]])
