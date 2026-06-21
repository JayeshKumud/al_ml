# %%
import pandas as pd #
import re #
import gensim
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim.corpora as corpora
from gensim.models import LdaModel

# LDA (Latent Dirichlet Allocation) is a popular topic 
# modeling technique used in Natural Language Processing (NLP) to discover hidden topics within a collection of documents.
# Instead of labeling documents manually, LDA automatically identifies groups of words that frequently occur together and treats them as topics.

data = pd.read_csv('../data/news_articles.csv')

articles = data['content']

# take just the content of the article, lowercase and remove punctuation
articles = articles.str.lower()
articles = articles.apply(lambda row: re.sub(r"[^\w\s]","",row))

# stop word removal
en_stopwords = stopwords.words('english')
articles = articles.apply(lambda row: ' '.join([word for word in row.split() if word not in en_stopwords]))

# tokenize
articles = articles.apply(lambda row: word_tokenize(row))

# stemming (done for speed as we have a lot of text)
stemmer = PorterStemmer()
articles = articles.apply(lambda row: [stemmer.stem(word) for word in row])

# Dictionary will assigned each word with unique id
dictionary = corpora.Dictionary(articles)

# Text has been vactorize: which word appear in each document
doc_term = [dictionary.doc2bow(text) for text in articles]

# create LDA model
num_topics = 2
lda_model = LdaModel(corpus=doc_term,
                                   id2word=dictionary,
                                   num_topics=num_topics)

lda_model.print_topics(num_topics=num_topics, num_words=5)
