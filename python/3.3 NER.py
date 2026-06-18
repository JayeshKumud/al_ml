# %%
from mistune import html
import spacy
from spacy import displacy
from spacy import tokenizer
from IPython.display import HTML, display
import pandas as pd
import re

nlp = spacy.load("en_core_web_sm")

# Read entire text from the CSV file and concatenate it into a single string for processing. 
google_text = open("../data/ner.txt", encoding="utf-8").read()
print(google_text)

spacy_doc = nlp(google_text)
for words in spacy_doc.ents:
    print(words.text, words.label_)


google_cleaned_text = re.sub(r'[^\w\s]', '', google_text).lower()
spacy_doc_cleaned = nlp(google_cleaned_text)

html = displacy.render(spacy_doc_cleaned, style="ent", jupyter=False)
# display(HTML(html))

with open("entities.html", "w", encoding="utf-8") as f:
    f.write(html)

import webbrowser
webbrowser.open("entities.html")
