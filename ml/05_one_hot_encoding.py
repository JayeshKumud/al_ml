# %%
# ---------------------------------------------------------
# Why One‑Hot Encoding?
# ---------------------------------------------------------
# Machine learning models cannot understand text labels like "new jersey" or "robbinsville".
# Linear Regression requires ALL inputs to be numeric.
#
# One‑Hot Encoding converts each category into separate 0/1 columns.
# Example:
#   town = "new jersey"  →  new jersey=1, west windsor=0, robbinsville=0
#
# We also drop ONE dummy column to avoid the "dummy variable trap"
# (perfect multicollinearity). This keeps the model stable.


import pandas as pd
from sklearn.linear_model import LinearRegression

# %%
# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
df = pd.read_csv("./data/05_one_hot_encoding_homeprices_town_type.csv")
# Dataset contains: town, area, price

df.head(10)  # Display important data frame


# %%
# ---------------------------------------------------------
# Convert categorical column "town" into dummy/one‑hot columns
# ---------------------------------------------------------

# Create dummy variables for each town
dummies = pd.get_dummies(df.town)

# Combine original dataframe with dummy columns
merged = pd.concat([df, dummies], axis="columns")

# Drop the original 'town' column because it's now encoded
final = merged.drop(["town"], axis="columns")

# Drop one dummy column to avoid multicollinearity
final = final.drop(["west windsor"], axis="columns")

final.head(10)  # Display encoded dataframe


# %%
# ---------------------------------------------------------
# Prepare features (X) and target (y)
# ---------------------------------------------------------
# X = all columns except price
x = final.drop("price", axis="columns")

# y = price column
y = final.price


# %%
# ---------------------------------------------------------
# Train Linear Regression Model
# ---------------------------------------------------------
# Linear Regression learns a relationship:
#
#   price = m1*area + m2*(new jersey) + m3*(robbinsville) + b
#
# where:
#   - m1, m2, m3 are coefficients (slopes)
#   - b is intercept (base price)
#
# The model finds the best-fitting line/plane that minimizes prediction error.

model = LinearRegression()
model.fit(x, y)

print("Model Coefficients (slopes):", model.coef_)
print("Model Intercept (base price):", model.intercept_)


# %%
# ---------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------
# Predict prices for the training data
predicted_prices = model.predict(x)
print("Predicted prices for training data:\n", predicted_prices)

# R² score (how well the model fits the data)
score = model.score(x, y)
print("Model Accuracy (R² score):", score)


# %%
# ---------------------------------------------------------
# Make predictions for new houses
# ---------------------------------------------------------
# IMPORTANT:
# Feature order must match training:
# [area, new jersey, robbinsville]

# 1. Predict price for 3400 sq ft home in west windsor
# west windsor was dropped → encoded as [area, 0, 0]
pred1 = model.predict([[3400, 0, 0]])
print("Price for 3400 sq ft home in west windsor:", pred1[0])

# 2. Predict price for 2800 sq ft home in robbinsville
# robbinsville = 1 → encoded as [area, 0, 1]
pred2 = model.predict([[2800, 0, 1]])
print("Price for 2800 sq ft home in robbinsville:", pred2[0])


# %%
# ---------------------------------------------------------
# Summary:
# ---------------------------------------------------------
# - One‑Hot Encoding converts text categories → numeric 0/1 columns.
# - Linear Regression learns how area + town affect house price.
# - Coefficients show how much each feature influences price.
# - R² score shows how well the model fits the data.
# - Predictions require

# %%
# ---------------------------------------------------------
# Label Encoding + OneHotEncoding Approach
# ---------------------------------------------------------
# In this approach:
# 1. LabelEncoder converts town → 0/1/2 (temporary numeric labels)
# 2. OneHotEncoder converts these labels → proper 0/1 dummy columns
#
# This is the sklearn pipeline-friendly way of doing one-hot encoding.

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# %%
# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
df = pd.read_csv("./data/05_one_hot_encoding_homeprices_town_type.csv")
df.head(10)  # Display important data frame


# %%
# ---------------------------------------------------------
# Step 1: Label Encode the 'town' column
# ---------------------------------------------------------
# LabelEncoder converts:
#   new jersey → 0
#   robbinsville → 1
#   west windsor → 2
#
# These numbers DO NOT represent real numeric meaning.
# They are only intermediate values before OneHotEncoding.

ln = LabelEncoder()
lbldf = df.copy()
lbldf["town"] = ln.fit_transform(lbldf["town"])

lbldf.head(10)


# %%
# ---------------------------------------------------------
# Step 2: Apply OneHotEncoder to the encoded 'town' column
# ---------------------------------------------------------
# OneHotEncoder converts:
#   0 → [1,0,0]
#   1 → [0,1,0]
#   2 → [0,0,1]
#
# This is equivalent to pd.get_dummies() but more flexible.

encoder = OneHotEncoder(sparse_output=False)
town_encoded = encoder.fit_transform(lbldf[["town"]])

print("OneHotEncoded town values:\n", town_encoded[:5])


# %%
# ---------------------------------------------------------
# Step 3: Build final training dataframe
# ---------------------------------------------------------
# Combine:
# - OneHotEncoded town columns
# - area column
# - price column

encoded_df = pd.DataFrame(town_encoded, columns=encoder.get_feature_names_out(["town"]))

final2 = pd.concat([encoded_df, lbldf["area"], lbldf["price"]], axis="columns")
final2.head(10)


# %%
# ---------------------------------------------------------
# Step 4: Prepare X and y
# ---------------------------------------------------------
X = final2.drop("price", axis="columns")
y = final2["price"]


# %%
# ---------------------------------------------------------
# Step 5: Train Linear Regression Model
# ---------------------------------------------------------
model2 = LinearRegression()
model2.fit(X, y)

print("Model Coefficients:", model2.coef_)
print("Model Intercept:", model2.intercept_)


# %%
# ---------------------------------------------------------
# Step 6: Model Evaluation
# ---------------------------------------------------------
predicted_prices2 = model2.predict(X)
print("Predicted prices:\n", predicted_prices2)

score2 = model2.score(X, y)
print("Model Accuracy (R² score):", score2)


# %%
# ---------------------------------------------------------
# Step 7: Predict same two homes
# ---------------------------------------------------------
# IMPORTANT:
# OneHotEncoder created columns in this order:
print("OneHotEncoder feature order:", encoder.get_feature_names_out(["town"]))

# Suppose LabelEncoder mapping was:
# ln.classes_ → ['new jersey', 'robbinsville', 'west windsor']
print("LabelEncoder classes:", ln.classes_)

# Meaning:
# new jersey → 0 → [1,0,0]
# robbinsville → 1 → [0,1,0]
# west windsor → 2 → [0,0,1]

# 1. Predict price for 3400 sq ft home in west windsor
# west windsor = [0,0,1]
pred_home1 = model2.predict([[0, 0, 1, 3400]])
print("Price for 3400 sq ft home in west windsor:", pred_home1[0])

# 2. Predict price for 2800 sq ft home in robbinsville
# robbinsville = [0,1,0]
pred_home2 = model2.predict([[0, 1, 0, 2800]])
print("Price for 2800 sq ft home in robbinsville:", pred_home2[0])
