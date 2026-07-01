# %%
# ------------------------------------------------------------
# MODEL & ALGORITHM DOCUMENTATION — GRIDSEARCHCV (DIGITS DATASET)
# ------------------------------------------------------------
# WHAT IS GRIDSEARCHCV?
#   GridSearchCV is an automated hyperparameter tuning technique.
#   It tries all combinations of parameters for a model and uses
#   cross‑validation to find:
#       • Best model
#       • Best hyperparameters
#       • Best accuracy score
#
# WHY WE USE IT:
#   • Manual tuning is slow and error‑prone
#   • GridSearchCV automates the process
#   • Ensures best model performance
#
# HOW GRIDSEARCHCV WORKS:
#   1. Define a model (e.g., SVM)
#   2. Define a parameter grid (e.g., C=[1,10], kernel=['rbf'])
#   3. GridSearchCV trains the model for every combination
#   4. Performs cross‑validation (cv=5)
#   5. Returns:
#        • best_score_
#        • best_params_
#        • full cv_results_
#
# PRINTED OUTPUTS:
#   • Best score for each model
#   • Best parameters for each model
#   • Final winner model
#
# ------------------------------------------------------------


from sklearn import datasets

digits = datasets.load_digits()

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier

# %%
# ------------------------------------------------------------
# DEFINE MODELS + THEIR PARAMETER GRIDS
# ------------------------------------------------------------
model_params = {
    "svm": {
        "model": svm.SVC(gamma="auto"),
        "params": {"C": [1, 10, 20], "kernel": ["rbf", "linear"]},
    },
    "random_forest": {
        "model": RandomForestClassifier(),
        "params": {"n_estimators": [1, 5, 10]},
    },
    "logistic_regression": {
        "model": LogisticRegression(solver="liblinear", multi_class="auto"),
        "params": {"C": [1, 5, 10]},
    },
    "naive_bayes_gaussian": {
        "model": GaussianNB(),
        "params": {},  # No hyperparameters to tune
    },
    "naive_bayes_multinomial": {
        "model": MultinomialNB(),
        "params": {},  # No hyperparameters to tune
    },
    "decision_tree": {
        "model": DecisionTreeClassifier(),
        "params": {
            "criterion": ["gini", "entropy"],
        },
    },
}

# %%
# ------------------------------------------------------------
# RUN GRIDSEARCHCV FOR EACH MODEL
# ------------------------------------------------------------
from sklearn.model_selection import GridSearchCV
import pandas as pd

scores = []

for model_name, mp in model_params.items():
    clf = GridSearchCV(mp["model"], mp["params"], cv=5, return_train_score=False)
    clf.fit(digits.data, digits.target)

    scores.append(
        {
            "model": model_name,
            "best_score": clf.best_score_,
            "best_params": clf.best_params_,
        }
    )

# %%
# ------------------------------------------------------------
# RESULTS DATAFRAME (IMPORTANT OUTPUT)
# ------------------------------------------------------------
df = pd.DataFrame(scores, columns=["model", "best_score", "best_params"])
df

# %%
# ------------------------------------------------------------
# FINAL CONCLUSION
# ------------------------------------------------------------
print("Best Model for Digits Classification:")
print(df.loc[df["best_score"].idxmax()])
