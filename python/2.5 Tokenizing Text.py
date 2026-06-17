# %%
import nltk 
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize, word_tokenize

sentences = "Her cat's name is Luna. Her dog's name is max"
print('sent_tokenize',sent_tokenize(sentences))
print('word_tokenize', word_tokenize(sentences))

sentence_2 = "Her cat's name is Luna and her dog's name is max"
print(word_tokenize(sentence_2))