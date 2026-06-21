# %%
import pandas as pd
import numpy as np
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn import metrics
from itertools import chain
from nltk import NaiveBayesClassifier

# No stemming, lemmetizing and stopwords as it can change sentiment completely

data = pd.read_csv('../data/book_reviews_sample.csv')


# Clean & Pre-process data

# Lower review text
data['reviewText_clean'] = data['reviewText'].str.lower()

# Remove punchuation
data['reviewText_clean'] = data.apply(lambda row: re.sub(r"[^\w\s]", "", row['reviewText_clean']), axis = 1)

# Rule base sentiment with Varder
varder_sentiment = SentimentIntensityAnalyzer()
data['vader_sentiment_score'] = data['reviewText_clean'].apply(lambda row: varder_sentiment.polarity_scores(row)['compound'])

# create labels
bins = [-1, -0.1, 0.1, 1]
names = ['negative', 'neutral', 'positive']
data['vader_sentiment_label'] = pd.cut(data['vader_sentiment_score'], bins, labels=names)

data['vader_sentiment_label'].value_counts().plot.bar()

# Sentiment using Trained Transfer Model
transformer_pipeline = pipeline("sentiment-analysis")
transformer_labels = []

for review in data['reviewText_clean'].values:
    sentiment_list = transformer_pipeline(review)
    sentiment_label = [sent['label'] for sent in sentiment_list]
    transformer_labels.append(sentiment_label)
    
data['transformer_sentiment_label'] = transformer_labels


data.head()
