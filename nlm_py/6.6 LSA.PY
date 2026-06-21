# %%
import pandas as pd #
import re #
import gensim
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim.corpora as corpora
from gensim.models import LsiModel
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt

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

# create LSA model
num_topics = 2
lsi_model = LsiModel(corpus=doc_term,
                                   id2word=dictionary,
                                   num_topics=num_topics)

lsi_model.print_topics(num_topics=num_topics, num_words=5)

# generate coherence scores to determine an optimum number of topics
coherence_values = []
model_list = []

min_topics = 2
max_topics = 11

for num_topics_i in range(min_topics, max_topics+1):
    model = LsiModel(doc_term, num_topics=num_topics_i, id2word = dictionary, random_seed=0)
    model_list.append(model)
    coherence_model = CoherenceModel(model=model, texts=articles, dictionary=dictionary, coherence='c_v')
    coherence_values.append(coherence_model.get_coherence())

plt.plot(range(min_topics, max_topics+1), coherence_values)
plt.xlabel("Number of Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


final_n_topics = 3
lsamodel_f = LsiModel(doc_term, num_topics=final_n_topics, id2word = dictionary) 
print(lsamodel_f.print_topics(num_topics=final_n_topics, num_words=5))