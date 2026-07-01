# %%
# ------------------------------------------------------------
# MODEL & ALGORITHM DOCUMENTATION — GRIDSEARCHCV
# ------------------------------------------------------------
# WHAT IS GRIDSEARCHCV?
#   GridSearchCV is a hyperparameter tuning technique that tries
#   all possible combinations of parameters for a given model.
#   It uses cross‑validation to evaluate each combination and
#   selects the best model + best hyperparameters.
#
# WHY WE USE IT:
#   • Manual tuning is slow and error‑prone
#   • GridSearchCV automates the process
#   • Ensures best model performance
#
# HOW GRIDSEARCHCV WORKS:
#   1. You define a model (e.g., SVM)
#   2. You define a parameter grid (e.g., C=[1,10], kernel=['rbf'])
#   3. GridSearchCV trains the model for every combination
#   4. Performs cross‑validation (cv=5)
#   5. Returns:
#        • best_score_
#        • best_params_
#        • full cv_results_
# ------------------------------------------------------------


from sklearn import svm, datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

# %%
# ------------------------------------------------------------
# LOAD IRIS DATASET
# ------------------------------------------------------------
iris = datasets.load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["flower"] = iris.target
df["flower"] = df["flower"].apply(lambda x: iris.target_names[x])

df[47:150]  # Important DataFrame slice

# %%
# ------------------------------------------------------------
# APPROACH 1 — MANUAL TUNING USING TRAIN/TEST SPLIT
# ------------------------------------------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3
)

model = svm.SVC(kernel="rbf", C=30, gamma="auto")
model.fit(X_train, y_train)

print("Manual Tuning Accuracy:", model.score(X_test, y_test))

# %%
# ------------------------------------------------------------
# APPROACH 2 — MANUAL TUNING USING CROSS‑VALIDATION
# ------------------------------------------------------------
print(
    "Linear Kernel, C=10:",
    cross_val_score(
        svm.SVC(kernel="linear", C=10, gamma="auto"), iris.data, iris.target, cv=5
    ),
)

print(
    "RBF Kernel, C=10:",
    cross_val_score(
        svm.SVC(kernel="rbf", C=10, gamma="auto"), iris.data, iris.target, cv=5
    ),
)

print(
    "RBF Kernel, C=20:",
    cross_val_score(
        svm.SVC(kernel="rbf", C=20, gamma="auto"), iris.data, iris.target, cv=5
    ),
)

# %%
# ------------------------------------------------------------
# APPROACH 2B — AUTOMATE MANUAL TUNING USING FOR LOOP
# ------------------------------------------------------------
kernels = ["rbf", "linear"]
C = [1, 10, 20]
avg_scores = {}

for kval in kernels:
    for cval in C:
        cv_scores = cross_val_score(
            svm.SVC(kernel=kval, C=cval, gamma="auto"), iris.data, iris.target, cv=5
        )
        avg_scores[kval + "_" + str(cval)] = np.average(cv_scores)

avg_scores  # Important DataFrame-like output

# %%
# ------------------------------------------------------------
# APPROACH 3 — GRIDSEARCHCV (Automated Hyperparameter Tuning)
# ------------------------------------------------------------
from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(
    svm.SVC(gamma="auto"),
    {"C": [1, 10, 20], "kernel": ["rbf", "linear"]},
    cv=5,
    return_train_score=False,
)

clf.fit(iris.data, iris.target)

# Full CV results
clf.cv_results_

# %%
# ------------------------------------------------------------
# CONVERT CV RESULTS TO DATAFRAME
# ------------------------------------------------------------
df = pd.DataFrame(clf.cv_results_)
df  # Important DataFrame

df[["param_C", "param_kernel", "mean_test_score"]]

print("Best Parameters:", clf.best_params_)
print("Best Score:", clf.best_score_)

# %%
# ------------------------------------------------------------
# APPROACH 4 — RANDOMIZEDSEARCHCV (Faster Tuning)
# ------------------------------------------------------------
from sklearn.model_selection import RandomizedSearchCV

rs = RandomizedSearchCV(
    svm.SVC(gamma="auto"),
    {"C": [1, 10, 20], "kernel": ["rbf", "linear"]},
    cv=5,
    return_train_score=False,
    n_iter=2,
)

rs.fit(iris.data, iris.target)

pd.DataFrame(rs.cv_results_)[["param_C", "param_kernel", "mean_test_score"]]

# %%
# ------------------------------------------------------------
# APPROACH 5 — MULTIPLE MODELS + MULTIPLE PARAMETERS
# ------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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
}

scores = []

for model_name, mp in model_params.items():
    clf = GridSearchCV(mp["model"], mp["params"], cv=5, return_train_score=False)
    clf.fit(iris.data, iris.target)
    scores.append(
        {
            "model": model_name,
            "best_score": clf.best_score_,
            "best_params": clf.best_params_,
        }
    )

df = pd.DataFrame(scores, columns=["model", "best_score", "best_params"])
df  # Important DataFrame

# %%
# ------------------------------------------------------------
# FINAL CONCLUSION
# ------------------------------------------------------------
print("Best Model for Iris Classification:")
print(df.loc[df["best_score"].idxmax()])
