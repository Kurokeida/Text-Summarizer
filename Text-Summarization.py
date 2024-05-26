import re
import nltk
import string
from goose3 import Goose
import heapq

#Using Goose function to retrieve and store a cleaned version of the link text
g = Goose()
url = 'Link here'
original_text = g.extract(url)
original_text = original_text.cleaned_text

original_text = re.sub(r'\s+', ' ', original_text)
stopwords = nltk.corpus.stopwords.words('english')

#function to remove stop words from the text
def preprocess(text):
    formatted_text = text.lower()
    tokens = []
    
    for token in nltk.word_tokenize(formatted_text):
        tokens.append(token)
        tokens = [word for word in tokens if word not in stopwords and word not in string.punctuation]
        formatted_text = ' '.join(element for element in tokens)

    
    return formatted_text

#Tokenizing each word in the text and getting the highest frequent text
formatted_text = preprocess(original_text)
word_frequency = nltk.FreqDist(nltk.word_tokenize(formatted_text))
highest_frequency = max(word_frequency.values())


#Weighting the values by highest frequency
for word in word_frequency.keys():
  #print(word)
  word_frequency[word] = (word_frequency[word] / highest_frequency)


sentence_list = nltk.sent_tokenize(original_text)

#Assigning a score to each tokenized sentence
score_sentences = {}
for sentence in sentence_list:
  #print(sentence)
  for word in nltk.word_tokenize(sentence.lower()):
    #print(word)
    if sentence not in score_sentences.keys():
      score_sentences[sentence] = word_frequency[word]
    else:
      score_sentences[sentence] += word_frequency[word]

best_sentences = heapq.nlargest(3, score_sentences, key = score_sentences.get)

summary = ' '.join(best_sentences)
print(summary)