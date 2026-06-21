sentence = "Her cat's name is Luna"
sentence = sentence.lower()
print(sentence)

sentence_list = ['Could you pass me the TV remote?', 
                 'It is IMPOSSIBLE to find this hotel', 
                 'Want to go for dinner on Tuesday?']
sentence_list = [sentence.lower() for sentence in sentence_list]
print(sentence_list)