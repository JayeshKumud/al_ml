import nltk
from nltk import ngrams
import pandas as pd
import matplotlib.pyplot as plt

tokens = ['the', 'rise', 'of', 'artificial', 'intelligence', 'has', 'led', 'to', 'significant', 'advancements', 'in', 'natural', 'language', 'processing', 'computer', 'vision', 'and', 'other', 'fields', 'machine', 'learning', 'algorithms', 'are', 'becoming', 'more', 'sophisticated', 'enabling', 'computers', 'to', 'perform', 'complex', 'tasks', 'that', 'were', 'once', 'thought', 'to', 'be', 'the', 'exclusive', 'domain', 'of', 'humans', 'with', 'the', 'advent', 'of', 'deep', 'learning', 'neural', 'networks', 'have', 'become', 'even', 'more', 'powerful', 'capable', 'of', 'processing', 'vast', 'amounts', 'of', 'data', 'and', 'learning', 'from', 'it', 'in', 'ways', 'that', 'were', 'not', 'possible', 'before', 'as', 'a', 'result', 'ai', 'is', 'increasingly', 'being', 'used', 'in', 'a', 'wide', 'range', 'of', 'industries', 'from', 'healthcare', 'to', 'finance', 'to', 'transportation', 'and', 'its', 'impact', 'is', 'only', 'set', 'to', 'grow', 'in', 'the', 'years', 'to', 'come']
series = pd.Series(tokens)
# print('Token counts:' + str(series.value_counts()))

unigrams = pd.Series(ngrams(tokens, 1)).value_counts()
print('Unigram counts:' + str(unigrams))

unigrams.plot.barh(color='blue', width=0.5, figsize=(10, 6), title='Unigram Counts')  
plt.title('Unigram Counts')

bigrams = pd.Series(ngrams(tokens, 2)).value_counts()
print('Bigram counts:' + str(bigrams))  

trigrams = pd.Series(ngrams(tokens, 3)).value_counts()
print('Trigram counts:' + str(trigrams))    