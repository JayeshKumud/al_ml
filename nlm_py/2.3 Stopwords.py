import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

en_stopwords = stopwords.words('english')
# print(en_stopwords)

sentence = "This is a sample sentence, showing off the stop words filtration."
sentence_nostop = ' '.join([word for word in sentence.split() if word.lower() not in en_stopwords]) 
print(sentence_nostop)