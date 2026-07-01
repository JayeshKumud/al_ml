# %%
# ------------------------------------------------------------
# MODEL & ALGORITHM DOCUMENTATION — NAIVE BAYES (MULTINOMIALNB)
# ------------------------------------------------------------
# MODEL NAME:
#   Multinomial Naive Bayes (MultinomialNB)
#
# MODEL DEFINITION:
#   Multinomial Naive Bayes is a probabilistic classifier based
#   on Bayes' Theorem. It is specifically designed for text data
#   where features represent word frequencies or counts.
#
# MODEL FUNCTIONING (HOW IT WORKS INTERNALLY):
#
#   1. Convert text into numerical features using CountVectorizer.
#        - Each email becomes a vector of word counts.
#
#   2. Compute prior probability of each class:
#        P(spam) and P(not spam)
#
#   3. Compute likelihood:
#        P(word | spam) and P(word | not spam)
#
#   4. Compute posterior probability:
#        P(spam | words in email)
#
#   5. Predict the class with highest probability.
#
# WHY MULTINOMIALNB IS USED HERE:
#   • Works extremely well for text classification
#   • Fast and efficient
#   • Handles high‑dimensional sparse data
#   • Ideal for spam detection
#
# FEATURES USED IN THIS CODE:
#   • Message text (converted to word count vectors)
# ------------------------------------------------------------


import pandas as pd

# %%
# ------------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------------
df = pd.read_csv("./data/14_naive_bayes_email_spam_filter")
df.head()

# %%
# ------------------------------------------------------------
# EXPLORE DATASET
# ------------------------------------------------------------
df.groupby("Category").describe()

# %%
# ------------------------------------------------------------
# CREATE TARGET COLUMN (spam = 1, ham = 0)
# ------------------------------------------------------------
df["spam"] = df["Category"].apply(lambda x: 1 if x == "spam" else 0)
df.head()

# %%
# ------------------------------------------------------------
# TRAIN-TEST SPLIT
# ------------------------------------------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam)

# %%
# ------------------------------------------------------------
# TEXT VECTORIZATION USING COUNT VECTORIZER
# ------------------------------------------------------------
from sklearn.feature_extraction.text import CountVectorizer

v = CountVectorizer()

# Fit on training data and transform
X_train_count = v.fit_transform(X_train.values)

# Display first 2 transformed rows
X_train_count.toarray()[:2]

# %%
# ------------------------------------------------------------
# TRAIN MULTINOMIAL NAIVE BAYES MODEL
# ------------------------------------------------------------
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train_count, y_train)

# %%
# ------------------------------------------------------------
# SAMPLE EMAIL PREDICTION
# ------------------------------------------------------------
emails = [
    "Hey mohan, can we get together to watch footbal game tomorrow?",
    "Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!",
]

emails_count = v.transform(emails)
model.predict(emails_count)

# %%
# ------------------------------------------------------------
# MODEL ACCURACY
# ------------------------------------------------------------
X_test_count = v.transform(X_test)
model.score(X_test_count, y_test)

# %%
# ------------------------------------------------------------
# PIPELINE (VECTORIZER + MODEL)
# ------------------------------------------------------------
from sklearn.pipeline import Pipeline

clf = Pipeline([("vectorizer", CountVectorizer()), ("nb", MultinomialNB())])

clf.fit(X_train, y_train)

# Pipeline accuracy
clf.score(X_test, y_test)

# Pipeline predictions
clf.predict(emails)
