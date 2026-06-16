import spacy
import pandas as pd

nlp = spacy.load("en_core_web_sm")

emma_ja =  open("pos_tagging.txt", encoding="utf-8").read()

spacy_doc = nlp(emma_ja)

pos_df = pd.DataFrame(columns=["token", "pos_tag"])
for token in spacy_doc:
    pos_df = pd.concat([pos_df, pd.DataFrame.from_records({"token": [token.text], "pos_tag": [token.pos_]})], ignore_index=True);
# print(pos_df.head(15))

pos_df_counts = pos_df.groupby(["token", "pos_tag"]).size().reset_index(name="count").sort_values(by="count", ascending=False)
# print(pos_df_counts.head(10))

pos_df_pos_counts = pos_df_counts.groupby(["pos_tag"])["token"].count().sort_values(ascending=False);
# print(pos_df_pos_counts.head(10))

nouns = pos_df_counts[pos_df_counts["pos_tag"] == "NOUN"].sort_values(by="count", ascending=False)
print(nouns.head(10))