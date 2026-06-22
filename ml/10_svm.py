# %%
# Generated from: 10_svm.ipynb
# This script demonstrates how Support Vector Machines (SVM) can be used
# for multiclass classification using the Iris dataset.
#
# SVM (Support Vector Machine):
# - A supervised ML algorithm that finds the best separating boundary (hyperplane)
# - Works well for both linear and non-linear classification using kernels

import pandas as pd
from sklearn.datasets import load_iris

# ---------------------------------------------------------
# Load Iris dataset
# ---------------------------------------------------------
iris = load_iris()

# Display feature and target names
print(iris.feature_names)
print(iris.target_names)

# %%
# ---------------------------------------------------------
# Create DataFrame from iris data
# ---------------------------------------------------------
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

# Display first 5 rows
df.head()

# Display samples for class 1 and class 2
df[df.target == 1].head()
df[df.target == 2].head()

# %%
# ---------------------------------------------------------
# Add readable flower names for better understanding
# ---------------------------------------------------------
df["flower_name"] = df.target.apply(lambda x: iris.target_names[x])
df.head()

# %%
# Display a slice of the dataset
df[45:55]

# ---------------------------------------------------------
# Split dataset by class (useful for plotting)
# ---------------------------------------------------------
df0 = df[:50]  # Setosa
df1 = df[50:100]  # Versicolor
df2 = df[100:]  # Virginica

# %%
df0.head()

# %%
df1.head()

# %%
df2.head()

# ---------------------------------------------------------
# Visualizations
# ---------------------------------------------------------
import matplotlib.pyplot as plt

# %%
# Plot Sepal Length vs Sepal Width for Setosa vs Versicolor
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.scatter(
    df0["sepal length (cm)"], df0["sepal width (cm)"], color="green", marker="+"
)
plt.scatter(df1["sepal length (cm)"], df1["sepal width (cm)"], color="blue", marker=".")

# %%
# Plot Petal Length vs Petal Width for Setosa vs Versicolor
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.scatter(
    df0["petal length (cm)"], df0["petal width (cm)"], color="green", marker="+"
)
plt.scatter(df1["petal length (cm)"], df1["petal width (cm)"], color="blue", marker=".")

# ---------------------------------------------------------
# Train SVM classifier
# ---------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Prepare features (x) and target (y)
x = df.drop(["target", "flower_name"], axis="columns")
y = df.target

# %%
# Display features
x.head()

# %%
# Display target values
y.head()

# Split into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# %%
len(x_train)

# %%
len(x_test)

# Train SVM model (default kernel = rbf)
model = SVC()
model.fit(x_train, y_train)

# %%
# Display model accuracy
print("Model accuracy:", model.score(x_test, y_test))

# %%
# Predict a sample flower
print(
    "Prediction for sample [4.8, 3.0, 1.5, 0.3]:", model.predict([[4.8, 3.0, 1.5, 0.3]])
)

# ---------------------------------------------------------
# Tune SVM parameters
# ---------------------------------------------------------

# %%
# 1. Regularization parameter C
model_C1 = SVC(C=1)
model_C1.fit(x_train, y_train)
print("Accuracy with C=1:", model_C1.score(x_test, y_test))

# %%
model_C10 = SVC(C=10)
model_C10.fit(x_train, y_train)
print("Accuracy with C=10:", model_C10.score(x_test, y_test))

# %%
# 2. Gamma parameter
model_g10 = SVC(gamma=10)
model_g10.fit(x_train, y_train)
print("Accuracy with gamma=10:", model_g10.score(x_test, y_test))

# %%
# 3. Kernel type
model_linear = SVC(kernel="linear")
model_linear.fit(x_train, y_train)
print("Accuracy with linear kernel:", model_linear.score(x_test, y_test))

# ---------------------------------------------------------
# Exercise
# ---------------------------------------------------------
# Train an SVM classifier using the sklearn digits dataset.
# Try different kernels (rbf, linear), tune C and gamma,
# and use 80% of samples for training.
