# %%
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF: Term frequency inverse document frequency
# TfidfVectorizer: Keep contaxt behind each words, converts text into numerical features using 
# TF-IDF scores instead of raw counts. It rewards words that are distinctive to a document 
# and penalizes words that appear everywhere.
# How important the word in document 

data = [' Most shark attacks occur about 10 feet from the beach since that is where the people are',
        'the efficiency with which he paired the socks in the drawer was quite admirable',
        'carol drank the blood as if she were a vampire',
        'giving directions that the mountains are to the west only works when you can see them',
        'the sign said there was road work ahead so he decided to speed up',
        'the gruff old man sat in the back of the bait shop grumbling to himself as he scooped out a handful of worms']

tfidfVectorizer = TfidfVectorizer()

tfidfVectorizer_fit = tfidfVectorizer.fit_transform(data)

tfidfbag = pd.DataFrame(tfidfVectorizer_fit.toarray(), columns=[tfidfVectorizer.get_feature_names_out()])

tfidfbag.head()
