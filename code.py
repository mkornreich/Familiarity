import os
import csv
from collections import defaultdict
import numpy as np

script_dir = os.path.dirname(__file__)

#Reads the document with word familiarity scores from MCR
columns = defaultdict(list)
with open(os.path.join(script_dir, "familiarity.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        for (k,v) in row.items():
            columns[k].append(v)

fam = columns["Familiarity"]
words = columns["Words"]

#Divides the words into categories of high and low familiarity
highfam = []
lowfam = []
for n in range(0, len(fam)):
    if int(fam[n]) < 300:
        lowfam += [words[n]]
    if int(fam[n]) > 600:
        highfam += [words[n]]

#Lowercases the words and removes duplicates
highfam = [x.lower() for x in list(set(highfam))]
lowfam = [x.lower() for x in list(set(lowfam))]

#Assigns categories to the words
data = {
  "High Familiarity": highfam,
  "Low Familiarity": lowfam,
}

#Puts each of the word into a category
categories = {word: key for key, words in data.items() for word in words}

#Loads all the word vectorizations from GloVe
#File can be obtained from http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
embeddings_index = {}
with open("glove.6B.100d.txt") as f:
  for line in f:
    values = line.split()
    word = values[0]
    embed = np.array(values[1:], dtype=np.float32)
    embeddings_index[word] = embed

#Processes the word embeddings
data_embeddings = {key: value for key, value in embeddings_index.items() if key in categories.keys()}

#Processes the familiarity score or a word
def process(query):
  query_embed = embeddings_index[query]
  scores = {}
  for word, embed in data_embeddings.items():
    category = categories[word]
    dist = query_embed.dot(embed)
    #calculates dot product between query's vector and the other word
    dist /= len(data[category])
    #divides the distance by the amount of words in that category
    scores[category] = scores.get(category, 0) + dist
    #adds the distance to the score for each category
  ewords = scores.get("High Familiarity")
  hwords = scores.get("Low Familiarity")
  return (ewords-hwords)
  #combines the score for the two categories

print(process("hello")) #May take up to a minute