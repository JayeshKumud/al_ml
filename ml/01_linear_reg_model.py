# %%
# %%
# Linear regression is one of the simplest and most important concepts in machine learning.
# It attempts to draw the best‑fit straight line through data points so we can predict
# a continuous value (e.g., house price) from an input feature (e.g., area).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Load the dataset
df = pd.read_csv("./data/01_linear_reg_model_homeprices.csv")
# Reads the CSV file into a pandas DataFrame.
# The dataset contains columns such as 'area' and 'price'.


# Plot the raw data
plt.xlabel("Area in Sq ft")
plt.ylabel("Price")
plt.scatter(df.area, df.price, color="red", marker="+")
# Creates a scatter plot to visualize the relationship between area (x-axis)
# and price (y-axis). This helps confirm that a linear trend exists.


# Create the Linear Regression model
reg = linear_model.LinearRegression()
# Initializes a LinearRegression model object that will learn slope (m) and intercept (b).


# Train (fit) the model
reg.fit(df[["area"]], df.price)
# df[["area"]] → 2D feature matrix required by scikit‑learn.
# df.price → target values.
# The model learns the best‑fit line using the equation:
#       y = m*x + b
#
# Where:
#   m = slope (model coefficient) → reg.coef_
#   b = intercept → reg.intercept_
#   x = area
#   y = predicted price


print("model coefficient", reg.coef_)
print("intercept", reg.intercept_)
# After training:
#   reg.coef_ contains the learned slope (m)
#   reg.intercept_ contains the learned intercept (b)


# Predict the price of a 3300 sq ft home
print(reg.predict([[3300]]))
# Input must be 2D → [[3300]]
# The model computes:
#       predicted_price = m * 3300 + b
