import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import pandas as pd


stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

data = pd.read_csv('tripadvisor_hotel_reviews.csv')  # Assuming you have a CSV file with text data
print(data.info())
# print(data.head())

data['review_to_lower'] = data['Review'].str.lower()  # Clean and lowercase text
# print(data.head())


en_stopwords = set(stopwords.words('english'))  # Get English stopwords
en_stopwords.remove('not')  # Remove 'not' from stopwords to keep negations
data['review_no_stopwords'] = data['review_to_lower'].apply(lambda x: ' '.join([word for word in x.split() if word not in en_stopwords]))  # Remove stopwords
# print(data.head())

data['review_no_stopwords_no_punk'] = data.apply(lambda x: x['review_no_stopwords'].sub("[*]", "star"), axis=1) # Remove punctuation and replace with "star"
data['review_no_stopwords_no_punk'] = data.apply(lambda x: re.sub(r'[^\w\s]', '', x['review_no_stopwords_no_punk']), axis=1)  # Remove punctuation
# print(data.head())

