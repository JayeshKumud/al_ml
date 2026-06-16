import pandas as pd
import matplotlib.pyplot as plt
import nltk
import re
import spacy
from spacy import displacy
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


bbc_data = pd.read_csv('bbc_news.csv')
# print(bbc_data.head())

title = pd.DataFrame([])