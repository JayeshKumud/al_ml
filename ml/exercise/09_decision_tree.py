# %%
# Decision Tree Classification on Titanic Dataset
# This script predicts passenger survival using a Decision Tree model.
# Each step is documented for clarity and reproducibility.

import pandas as pd
import math
from IPython.display import display
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# %%
# Load dataset
df = pd.read_csv("../data/09_decision_tree_model_titanic.csv")

# %%
# Handle missing Age values using median (robust to outliers)
age_mean = math.floor(df["Age"].median())
df["Age"].fillna(age_mean, inplace=True)

# %%
# Encode Sex column (male=1, female=0)
encoder = LabelEncoder()
df["Sex"] = encoder.fit_transform(df["Sex"])

# %%
# Select input features by dropping irrelevant columns
# Features used:
#   - Pclass: Passenger class (1, 2, 3)
#   - Age: Passenger age
#   - Fare: Ticket fare
#   - Sex: Encoded gender (0=female, 1=male)
input = df.drop(
    [
        "Survived",
        "PassengerId",
        "Name",
        "SibSp",
        "Parch",
        "Ticket",
        "Embarked",
        "Cabin",
    ],
    axis="columns",
)

# %%
# Display input features
display(input)

# %%
# Target column (1 = survived, 0 = died)
target = df["Survived"]
display(target)

# %%
# Split dataset into training and testing sets
# 80% training, 20% testing
x_train, x_test, y_train, y_test = train_test_split(
    input, target, train_size=0.8, random_state=10
)

# %%
# Initialize Decision Tree Classifier
# Decision Tree works by splitting data based on feature thresholds
# until it learns patterns that separate classes (survived vs not survived)
model = DecisionTreeClassifier()

# Train the model
model.fit(x_train, y_train)

# %%
# Predict on test dataset
predicted = model.predict(x_test)
print("Predicted values on test set:")
print(predicted)

# %%
# Model accuracy score
score = model.score(x_test, y_test)
print("Model Accuracy:", score)

# %%
# Predict survival for random passenger data
# Format: [Pclass, Age, Fare, Sex]
# Example: 3rd class, age 25, fare 8.50, male (1)
random_passenger = [[3, 25, 8.50, 1]]

random_prediction = model.predict(random_passenger)
print("\nRandom Passenger Prediction:", random_prediction)
print("1 = Survived, 0 = Did Not Survive")
