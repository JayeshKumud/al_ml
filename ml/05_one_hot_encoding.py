# %%
# When a dataset contains text or categorical features (like 'town'), they must be converted
# into numeric form before training a model. We use dummy variables (one‑hot encoding) to turn
# each category into separate 0/1 columns so the model can understand and learn from them.


import pandas as pd

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
df = pd.read_csv("./data/05_one_hot_encoding_homeprices_town_type.csv")
# Dataset contains: town, area, price


# ---------------------------------------------------------
# Convert categorical column "town" into dummy/one‑hot columns
# ---------------------------------------------------------

# Create dummy variables for each town
# Example: 'new jersey', 'west windsor', 'robbinsville'
dummies = pd.get_dummies(df.town)

# Combine original dataframe with dummy columns
merged = pd.concat([df, dummies], axis="columns")

# Drop the original 'town' column because it's now encoded
final = merged.drop(["town"], axis="columns")

# Drop one dummy column to avoid the "dummy variable trap"
# (removing one category prevents multicollinearity)
final = final.drop(["west windsor"], axis="columns")


# ---------------------------------------------------------
# Prepare features (X) and target (y)
# ---------------------------------------------------------

# X = all columns except price
x = final.drop("price", axis="columns")

# y = price column
y = final.price


# ---------------------------------------------------------
# Train Linear Regression Model
# ---------------------------------------------------------
from sklearn.linear_model import LinearRegression

model = LinearRegression()

# Fit the model using X and y
model.fit(x, y)


# ---------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------

# Predict prices for the training data
model.predict(x)  # Example: 2600 sq ft home in New Jersey

# R² score (how well the model fits the data)
model.score(x, y)


# ---------------------------------------------------------
# Make predictions for new houses
# ---------------------------------------------------------

# Predict price for:
# 3400 sq ft home in west windsor
# Encoding: [area, new jersey, robbinsville]
model.predict([[3400, 0, 0]])

# Predict price for:
# 2800 sq ft home in robbinsville
model.predict([[2800, 0, 1]])


# ---------------------------------------------------------
# Note:
# The same one‑hot encoding can be done using
# sklearn.preprocessing.OneHotEncoder
# ---------------------------------------------------------
