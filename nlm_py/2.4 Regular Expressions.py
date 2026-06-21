import re
path = r"c\Desttop\notes.txt"
print("Path:", path)

result_search_found = re.search("pattern", "pattern exists in this string")
print("Search result:", result_search_found)

result_search_not_found = re.search("pattern", "no match here")
print("Search result:", result_search_not_found)

string = "Sarah has a cat. Her cat's name is Luna."
new_string = re.sub("cat", "dog", string)
print("Original string:", string)
print("New string:", new_string)


customer_reviews = ['sam was a great help to me in the store', 
                    'the cashier was very rude to me, I think her name was eleanor', 
                    'amazing work from sadeen!', 
                    'sarah was able to help me find the items i needed quickly', 
                    'lucy is such a great addition to the team', 
                    'great service from sara she found me what i wanted'
                   ]

# Optional latter: Get list of review from Sarah, keep in mind that name can be misspelled as sarah in the reviews. Use regex to find all reviews that contain the name "Sarah" or any variation of it (e.g., "sarah", "Sara", "sara", etc.).
pattern_to_find = r"sarah?"
sarah_review = [review for review in customer_reviews if re.search(pattern_to_find, review)]
print("Reviews mentioning Sarah:", sarah_review)

# Start with a letter: Get list of review that start with a letter. Use regex to find all reviews that start with a letter (a-z or A-Z).
pattern_to_find = r"^a"
review_start_with_a = [review for review in customer_reviews if re.search(pattern_to_find, review)]
print("Reviews starting with 'a':", review_start_with_a)

# End with a letter: Get list of review that end with a letter. Use regex to find all reviews that end with a letter (a-z or A-Z). 
pattern_to_find = r"y$"
reviews_end_with_y = [review for review in customer_reviews if re.search(pattern_to_find, review)]
print("Reviews ending with 'y':", reviews_end_with_y)

# Either this or that: Get list of review that contain either "need" or "wanted". Use regex to find all reviews that contain either of these variations.
pattern_to_find = r"(need|want)ed"  
review_contain_need_or_want = [review for review in customer_reviews if re.search(pattern_to_find, review)]
print("Reviews containing 'need' or 'wanted':", review_contain_need_or_want)

# No punctuation: Remove all punctuation from the reviews and get list of reviews without punctuation (e.g., ., !, ?, etc.).
pattern_to_find = r"[^\w\s]"
review_with_no_punctuation = [re.sub(pattern_to_find, "", review) for review in customer_reviews] 
print("Reviews without punctuation:", review_with_no_punctuation)
