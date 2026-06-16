import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
import re
import pandas as pd


stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Load the csv dataset
data = pd.read_csv('tripadvisor_hotel_reviews.csv')
# print(data.info())
# print(data.head())

# To lowercase the text and clean it
data['review_to_lower'] = data['Review'].str.lower()  # Clean and lowercase text
# print(data.head())

# To remove stopwords from the text
en_stopwords = set(stopwords.words('english'))  # Get English stopwords
en_stopwords.remove('not')  # Remove 'not' from stopwords to keep negations
data['review_no_stopwords'] = data['review_to_lower'].apply(lambda x: ' '.join([word for word in x.split() if word not in en_stopwords]))  # Remove stopwords
# print(data.head())

# To remove punctuation and replace with "star"
data['review_no_stopwords_no_punk'] = data.apply(lambda x: re.sub("[*]", "star", x['review_no_stopwords']), axis=1) # Remove punctuation and replace with "star"
data['review_no_stopwords_no_punk'] = data.apply(lambda x: re.sub(r'[^\w\s]', '', x['review_no_stopwords_no_punk']), axis=1)  # Remove punctuation
# print(data.head())

# Tokenize the text
data['review_tokenize'] = data['review_no_stopwords_no_punk'].apply(lambda x: word_tokenize(x))  # Tokenize text
# print(data.head())

# To apply stemming
data['review_stammed'] = data['review_tokenize'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x]))  # Apply stemming
# print(data.head())

# To apply lemmatization
data['Review_lemmatized'] = data['review_tokenize'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x]))  # Apply lemmatization
# print(data.head())

# Sum the tokenized reviews into a single list for n-gram analysis
token_clean = sum(data['review_tokenize'], [])

# Get the frequency of unigrams
unigrams = pd.Series(ngrams(token_clean, 1)).value_counts()
print(unigrams)

# Get the frequency of bigrams
bigrams = pd.Series(ngrams(token_clean, 2)).value_counts()
print(bigrams)