# %%
# Sentiment Analysis: Sentiment analysis is an NLP task that identifies the emotional tone of text 
# typically classifying it as positive, negative, or neutral (and sometimes more fine-grained like joy, anger, fear, etc.).

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# %%
sentence_1 = "i had a great time at the movie it was really funny"
sentence_2 = "i had a great time at the movie but the parking was terrible"
sentence_3 = "i had a great time at the movie but the parking wasn't great"
sentence_4 = "i went to see a movie"

# %%
# Sentiment analysis using TextBlob
sentiment_sentence_1 = TextBlob(sentence_1).polarity
print(sentence_1, ':', sentiment_sentence_1)

sentiment_sentence_2 = TextBlob(sentence_2).polarity
print(sentence_2, ':', sentiment_sentence_2)

sentiment_sentence_3 = TextBlob(sentence_3).polarity
print(sentence_3, ':', sentiment_sentence_3)

sentiment_sentence_4 = TextBlob(sentence_4).polarity
print(sentence_4, ':', sentiment_sentence_4)

# %%
# Sentiment analysis using vaderSentiment
# It returned dictionary of score neutral, negative, positive and compound
# Compound score is overall sentimental value 
sentimentAnalyzer = SentimentIntensityAnalyzer()

sentiment_sentence_1 = sentimentAnalyzer.polarity_scores(sentence_1)
print(sentence_1, '\n', sentiment_sentence_1)

sentiment_sentence_2 = sentimentAnalyzer.polarity_scores(sentence_1)
print(sentence_2, '\n', sentiment_sentence_2)

sentiment_sentence_1 = sentimentAnalyzer.polarity_scores(sentence_1)
print(sentence_3, '\n', sentiment_sentence_3)

sentiment_sentence_1 = sentimentAnalyzer.polarity_scores(sentence_1)
print(sentence_4, '\n', sentiment_sentence_4)
