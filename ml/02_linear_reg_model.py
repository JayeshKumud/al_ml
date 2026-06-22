# %%
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D  # Enables 3D plotting

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
df = pd.read_csv("./data/02_linear_reg_model_homeprices_multi_feature.csv")
# The CSV contains: area, bedrooms, age, price


# ---------------------------------------------------------
# Data Pre‑processing
# ---------------------------------------------------------

# Compute the median number of bedrooms
# (Used to fill missing values)
median_bedrooms = math.floor(df.bedrooms.median())
print(median_bedrooms)

# Replace NaN bedrooms with the median value
df.bedrooms.fillna(median_bedrooms, inplace=True)


# ---------------------------------------------------------
# Train Linear Regression Model
# ---------------------------------------------------------

# Create a LinearRegression model object
reg = linear_model.LinearRegression()

# Fit the model using 3 features:
# area, bedrooms, age → predict price
reg.fit(df[["area", "bedrooms", "age"]], df.price)

# Print learned coefficients (slopes for each feature)
print("model coefficient", reg.coef_)

# Print intercept (b value)
print("intercept", reg.intercept_)

# Predict price for a new house:
# area=3000, bedrooms=3, age=40
print("predict", reg.predict([[3000, 3, 40]]))


# ---------------------------------------------------------
# 3D Scatter Plot (Area vs Bedrooms vs Price)
# ---------------------------------------------------------

# Create a new figure
fig = plt.figure()

# Add a 3D subplot
ax = fig.add_subplot(111, projection="3d")

# Plot the data points in 3D space
ax.scatter(df.area, df.bedrooms, df.price, color="red")

# Label axes
ax.set_xlabel("Area")
ax.set_ylabel("Bedrooms")
ax.set_zlabel("Price")

# Display the plot
plt.show()
