# %%
# ---------------------------------------------------------
# Logistic regression is a machine‑learning algorithm used to predict a yes/no outcome — also called a binary classification problem.
# 1. Import required libraries
# ---------------------------------------------------------
import pandas as pd
from matplotlib import pyplot as plt

# ---------------------------------------------------------
# 2. Load dataset
# ---------------------------------------------------------
df = pd.read_csv("./data/07_logistic_regression_insurance.csv")
df.head()  # Shows first 5 rows for inspection

# ---------------------------------------------------------
# 3. Plot raw data (Age vs Bought Insurance)
# ---------------------------------------------------------
plt.scatter(df.age, df.bought_insurance, marker="+", color="red")
plt.xlabel("Age")
plt.ylabel("Bought Insurance (0/1)")
plt.title("Insurance Purchase vs Age")
plt.show()

# ---------------------------------------------------------
# 4. Split data into training and testing sets
# ---------------------------------------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df[["age"]], df.bought_insurance, train_size=0.8
)

# ---------------------------------------------------------
# 5. Create and train Logistic Regression model
# ---------------------------------------------------------
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

# ---------------------------------------------------------
# 6. Inspect test data
# ---------------------------------------------------------
print("X_test values:")
print(X_test)

# ---------------------------------------------------------
# 7. Predict using the trained model
# ---------------------------------------------------------
y_predicted = model.predict(X_test)
print("Predicted labels:")
print(y_predicted)

# ---------------------------------------------------------
# 8. Probability predictions (sigmoid output)
# ---------------------------------------------------------
print("Prediction probabilities:")
print(model.predict_proba(X_test))

# ---------------------------------------------------------
# 9. Model accuracy score
# ---------------------------------------------------------
print("Model accuracy:", model.score(X_test, y_test))

# ---------------------------------------------------------
# 10. Model parameters (slope and intercept)
# ---------------------------------------------------------
print("Model coefficient:", model.coef_)
print("Model intercept:", model.intercept_)

# ---------------------------------------------------------
# 11. Manual sigmoid + prediction function
# ---------------------------------------------------------
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Using approximate learned values:
# coef_ ≈ 0.0415 → rounded to 0.042
# intercept_ ≈ -1.527 → rounded to -1.53
def prediction_function(age):
    z = 0.042 * age - 1.53
    y = sigmoid(z)
    return y


# ---------------------------------------------------------
# 12. Manual predictions with age printed first
# ---------------------------------------------------------
age = 35
print("Age:", age, "Predicted probability:", prediction_function(age))

age = 43
print("Age:", age, "Predicted probability:", prediction_function(age))
