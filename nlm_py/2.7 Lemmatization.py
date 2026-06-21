# %%
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()    

connect_tokens = ['connecting', 'connected', 'connectivity', 'connect', 'connects']
learn_tokens = ['learned', 'learning', 'learn', 'learns', 'learner', 'learners']
likes_tokens = ['likes', 'better', 'worse']

print([lemmatizer.lemmatize(token) for token in connect_tokens])
print([lemmatizer.lemmatize(token) for token in learn_tokens])
print([lemmatizer.lemmatize(token) for token in likes_tokens])