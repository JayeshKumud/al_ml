# %%
import numpy as np
import joblib  # modern import (sklearn.externals.joblib is deprecated)

# ---------------------------------------------------------
# Gradient Descent for Simple Linear Regression (y = mx + b)
# ---------------------------------------------------------
# Goal:
# Adjust slope (m) and intercept (b) to minimize MSE (mean squared error)
# between predicted values and actual values.
# ---------------------------------------------------------


def gradient_descent(x, y):
    m_curr = 0  # initial slope
    b_curr = 0  # initial intercept
    iterations = 10000  # number of update cycles
    n = len(x)  # number of samples
    learning_rate = 0.08

    for i in range(iterations):

        # Predicted values using current m and b
        y_predicted = m_curr * x + b_curr

        # Cost function (MSE)
        cost = (1 / n) * sum((y - y_predicted) ** 2)

        # Gradient of cost w.r.t m
        md = -(2 / n) * sum(x * (y - y_predicted))

        # Gradient of cost w.r.t b
        bd = -(2 / n) * sum(y - y_predicted)

        # Update parameters
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd

        # Debug print for each iteration
        print(f"m {m_curr}, b {b_curr}, cost {cost}, iteration {i}")

    # Return final learned parameters
    return m_curr, b_curr


# ---------------------------------------------------------
# Training Data
# ---------------------------------------------------------
x = np.array([1, 2, 3, 4, 5])  # feature values
y = np.array([5, 7, 9, 11, 13])  # target values

# Train model → get slope and intercept
m, b = gradient_descent(x, y)


# ---------------------------------------------------------
# Save the trained model parameters
# ---------------------------------------------------------
# Save as a tuple (m, b)
joblib.dump((m, b), "./trained_model/gd_linear_model.pkl")


# ---------------------------------------------------------
# Load the saved model
# ---------------------------------------------------------
m_loaded, b_loaded = joblib.load("./trained_model/gd_linear_model.pkl")


# ---------------------------------------------------------
# Use loaded model for prediction
# ---------------------------------------------------------
def predict(x):
    return m_loaded * x + b_loaded


print("Prediction for x=5000:", predict(15))
