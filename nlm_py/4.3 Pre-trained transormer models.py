# %%
# Transformer-Based Sentiment Analysis: performing sentiment analysis using Transformer neural network
# Transformers (BERT, RoBERTa, DistilBERT, etc.) use self-attention to understand relationships between all words in a sentence simultaneously.
# They are pre-trained on massive corpora and then fine-tuned on sentiment datasets.
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")

sentence_1 = "i had a great time at the movie it was really funny"
sentence_2 = "i had a great time at the movie but the parking was terrible"
sentence_3 = "i had a great time at the movie but the parking wasn't great"
sentence_4 = "i went to see a movie"

print(sentence_1, ':', sentiment_pipeline(sentence_1))
print(sentence_2, ':', sentiment_pipeline(sentence_2))
print(sentence_3, ':', sentiment_pipeline(sentence_3))
print(sentence_4, ':', sentiment_pipeline(sentence_4))

sentiment_pipeline_model = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
print(sentence_1, ':', sentiment_pipeline_model(sentence_1))
print(sentence_2, ':', sentiment_pipeline_model(sentence_2))
print(sentence_3, ':', sentiment_pipeline_model(sentence_3))
print(sentence_4, ':', sentiment_pipeline_model(sentence_4))