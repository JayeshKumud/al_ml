from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
connect_tokens = ['connecting', 'connected', 'connectivity', 'connect', 'connects']
print([stemmer.stem(token) for token in connect_tokens])

learn_tokens = ['learned', 'learning', 'learn', 'learns', 'learner', 'learners']
print([stemmer.stem(token) for token in learn_tokens])