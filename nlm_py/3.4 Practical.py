# %%
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import re
import spacy
from spacy import displacy
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

bbc_data = pd.read_csv('../data/bbc_news.csv')

# Get the title only
title = pd.DataFrame(bbc_data['title'])

# Convert title to lowercase
title['title_lower'] = title['title'].str.lower()

# Remove stopwords
nltk.download('stopwords')
en_stopwords = stopwords.words('english')
title['title_no_stopwords'] = title['title_lower'].apply(lambda x: ' '.join([word for word in x.split() if word not in en_stopwords]))

# Remove punctuation
title['title_no_punctuation'] = title['title_no_stopwords'].apply(lambda x: re.sub(r"[^\w\s]","",x))

# Tokanize
title['title_tokanize'] = title.apply(lambda x: word_tokenize(x['title_no_punctuation']), axis=1)

# Lamatize
lemmatizer = WordNetLemmatizer()
# Lemmatize each token in the tokenized title (title_tokanize is a list of tokens)
title['title_lemmatizer'] = title['title_tokanize'].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])
# title["title_lemmatizer"] = title.apply(lambda row: [lemmatizer.lemmatize(word) for word in row["title"]],axis=1)

token_raw = sum(title['title_lemmatizer'], [])
nlp = spacy.load('en_core_web_sm')
spacy_doc = nlp(' '.join(token_raw))

pos_df = pd.DataFrame(columns=['token','pos_tag'])

for token in spacy_doc:
    pos_df = pd.concat([pos_df, pd.DataFrame.from_records({"token": [token.text], "pos_tag": [token.pos_]})], ignore_index=True);

pos_df_counts = pos_df.groupby(["token", "pos_tag"]).size().reset_index(name="count").sort_values(by="count", ascending=False)

nouns = pos_df_counts[pos_df_counts["pos_tag"] == "NOUN"].sort_values(by="count", ascending=False)

ner_df = pd.DataFrame(columns=['token', 'ner_tag'])
for token in spacy_doc.ents:
    if pd.isna(token.label_) is False:
        ner_df = pd.concat([ner_df, pd.DataFrame.from_records({"token": [token.text], "ner_tag": [token.label_]})], ignore_index=True);

ner_df_counts = ner_df.groupby(["token", "ner_tag"]).size().reset_index(name="count").sort_values(by="count", ascending=False)
ner_df_counts.head()

