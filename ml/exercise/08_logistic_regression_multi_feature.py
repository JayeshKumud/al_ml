# %%
# ------------------------------------------------------------
# Iris Dataset – Logistic Regression Classification
# ------------------------------------------------------------
# This script loads the Iris dataset, performs exploratory steps,
# visualizes the three flower classes, prepares features and labels,
# trains a Logistic Regression model, and evaluates its accuracy.
# ------------------------------------------------------------

import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# %%
# ------------------------------------------------------------
# 1. Load the Iris dataset
# ------------------------------------------------------------
iris = load_iris()

display("Directory", dir(iris))
display("target", iris["target"])
display("feature_names", iris["feature_names"])
display("target_names", iris["target_names"])

# %%
# ------------------------------------------------------------
# 2. Create DataFrame from iris.data
# ------------------------------------------------------------
df = pd.DataFrame(iris.data, columns=iris.feature_names)
display(df)

# Add numeric target column
df["target"] = iris.target

# Map numeric target → flower name
df["flower_name"] = df["target"].apply(lambda x: iris.target_names[x])

# Display full DataFrame
display(df)

# %%
# ------------------------------------------------------------
# 3. Filter each flower type into separate DataFrames
# ------------------------------------------------------------
setosa_df = df[df["flower_name"] == "setosa"]
versicolor_df = df[df["flower_name"] == "versicolor"]
virginica_df = df[df["flower_name"] == "virginica"]

# %%
# ------------------------------------------------------------
# 4. Visualization – Sepal Dimensions
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Sepal Dimensions by Flower Type")

plt.scatter(
    setosa_df["sepal length (cm)"],
    setosa_df["sepal width (cm)"],
    color="green",
    marker="+",
    label="Setosa",
)
plt.scatter(
    versicolor_df["sepal length (cm)"],
    versicolor_df["sepal width (cm)"],
    color="blue",
    marker="+",
    label="Versicolor",
)
plt.scatter(
    virginica_df["sepal length (cm)"],
    virginica_df["sepal width (cm)"],
    color="red",
    marker="+",
    label="Virginica",
)

plt.legend()
plt.show()

# %%
# ------------------------------------------------------------
# 5. Visualization – Petal Dimensions
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Petal Dimensions by Flower Type")

plt.scatter(
    setosa_df["petal length (cm)"],
    setosa_df["petal width (cm)"],
    color="green",
    marker="+",
    label="Setosa",
)
plt.scatter(
    versicolor_df["petal length (cm)"],
    versicolor_df["petal width (cm)"],
    color="blue",
    marker="+",
    label="Versicolor",
)
plt.scatter(
    virginica_df["petal length (cm)"],
    virginica_df["petal width (cm)"],
    color="red",
    marker="+",
    label="Virginica",
)

plt.legend()
plt.show()

# %%
# ------------------------------------------------------------
# 6. Prepare Feature Matrix (X) and Target Vector (y)
# ------------------------------------------------------------
# X = all numeric feature columns
# y = numeric target labels (0, 1, 2)
X = df.drop(["target", "flower_name"], axis="columns")
y = df["target"]

display("Feature Matrix (X)", X.head())
display("Target Vector (y)", y.head())

# %%
# ------------------------------------------------------------
# 7. Train-Test Split
# ------------------------------------------------------------
# 80% training, 20% testing
x_train, x_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=10
)

print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))

# %%
# ------------------------------------------------------------
# 8. Logistic Regression Model
# ------------------------------------------------------------
# Logistic Regression is a classification algorithm that predicts
# the probability of each flower class (Setosa, Versicolor, Virginica).
# It works well for multi-class classification using the softmax function.
# ------------------------------------------------------------

rg_model = LogisticRegression(max_iter=200)
rg_model.fit(x_train, y_train)

# %%
# ------------------------------------------------------------
# 9. Model Evaluation
# ------------------------------------------------------------
accuracy = rg_model.score(x_test, y_test)
print("Model Accuracy:", accuracy)

# Print predictions for test set
predicted_values = rg_model.predict(x_test)
print("Predicted Classes:", predicted_values)

# Print actual values for comparison
print("Actual Classes:", list(y_test))
