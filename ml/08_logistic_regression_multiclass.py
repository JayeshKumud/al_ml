# %%
# %%
# Generated from: 8_logistic_regression_multiclass.ipynb
# This script demonstrates how logistic regression can be used for
# multiclass classification using the sklearn digits dataset.

# ---------------------------------------------------------
# Load and explore the digits dataset
# ---------------------------------------------------------
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()  # Contains 8×8 pixel images of handwritten digits (0–9)

# Display first few digit images
plt.gray()
for i in range(5):
    plt.matshow(digits.images[i])  # Visualize raw pixel data

# digits.data → flattened pixel values (64 features per image)
# digits.target → actual digit labels (0–9)
digits.data[0]


# ---------------------------------------------------------
# Create and train logistic regression model
# ---------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

model = LogisticRegression(max_iter=2000)
# max_iter increased to ensure convergence on this dataset

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2
)

# Train the model on handwritten digit images
model.fit(X_train, y_train)


# ---------------------------------------------------------
# Evaluate model accuracy
# ---------------------------------------------------------
print("Returns accuracy on unseen test data", model.score(X_test, y_test))
# Returns accuracy on unseen test data

print("Predict labels for first 5 samples", model.predict(digits.data[0:5]))
# Predict labels for first 5 samples


# ---------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------
y_predicted = model.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_predicted)
cm  # Raw confusion matrix values

# Visualize confusion matrix as heatmap
import seaborn as sn

plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")


# ---------------------------------------------------------
# Exercise
# ---------------------------------------------------------
# Use sklearn.datasets iris dataset to train a logistic regression model.
# The iris dataset contains 150 samples with 4 features:
#   1. Sepal Length
#   2. Sepal Width
#   3. Petal Length
#   4. Petal Width
#
# Your task:
#   - Train logistic regression on the iris dataset
#   - Measure accuracy
#   - Predict flower species for test samples
#
# Iris classes:
#   1. Setosa
#   2. Versicolour
#   3. Virginica
#
# (Image reference: iris_petal_sepal.png)
