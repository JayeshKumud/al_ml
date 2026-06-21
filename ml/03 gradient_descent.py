# %%
import numpy as np
import joblib  # sklearn.externals.joblib is deprecated

# ---------------------------------------------------------
# Gradient Descent for Simple Linear Regression (y = mx + b)
# ---------------------------------------------------------
# Goal:
# Adjust slope (m) and intercept (b) step-by-step to minimize
# the Mean Squared Error (MSE) between predicted and actual values.
# ---------------------------------------------------------


def gradient_descent(x, y):
    m_curr = 0  # initial slope
    b_curr = 0  # initial intercept
    iterations = 10000  # number of update cycles
    n = len(x)  # number of data points
    learning_rate = 0.08

    for i in range(iterations):

        # Predicted values using current m and b
        y_predicted = m_curr * x + b_curr

        # Cost function (MSE): average of squared errors
        cost = (1 / n) * sum([val**2 for val in (y - y_predicted)])

        # Partial derivative of cost w.r.t m
        md = -(2 / n) * sum(x * (y - y_predicted))

        # Partial derivative of cost w.r.t b
        bd = -(2 / n) * sum(y - y_predicted)

        # Update m and b using gradient descent rule
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd

        # Print progress for each iteration
        print("m {}, b {}, cost {} iteration {}".format(m_curr, b_curr, cost, i))

    # Return final learned parameters
    return m_curr, b_curr


# ---------------------------------------------------------
# Training Data
# ---------------------------------------------------------
x = np.array([1, 2, 3, 4, 5])  # input feature
y = np.array([5, 7, 9, 11, 13])  # target values

# Run gradient descent
m, b = gradient_descent(x, y)

# ---------------------------------------------------------
# Save model parameters using joblib
# ---------------------------------------------------------
# Save (m, b) tuple into a file
joblib.dump((m, b), "model_joblib.pkl")
