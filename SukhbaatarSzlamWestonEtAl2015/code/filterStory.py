import sys
import json
import os
import numpy as np
import re
from collections import Counter
#import nltk
from pyemd import emd
#sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
def avg_feature_vector(words, model, num_features):
	#function to average all words vectors in a given paragraph
	featureVec = np.zeros((num_features,), dtype="float32")
	nwords = 0
	#list containing names of words in the vocabulary
	for word in words:
		if word in model:
				nwords = nwords+1
				featureVec = np.add(featureVec, model[word])
	if(nwords>0):
		featureVec = np.divide(featureVec, nwords)
	return featureVec

read_from = "../data/wiki/test/test-00000-of-00015.json"
write_to = "../data/wiki/test/filtered-test-00000-of-00015.json"
EMBED_DIR = '/home/ashwin/WorkBench/NLP/QA/data/Embed'
EMBED_NAME = 'wiki.en.vec'
EMBED_NAME = 'glove.6B.300d.txt'
print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(EMBED_DIR, EMBED_NAME))
for line in f:
    values = line.split()
    if (len(values) != 301):
        continue
    word = values[0].lower()
    coefs = np.asarray(values[1:], dtype='float64')
    embeddings_index[word] = coefs
f.close()

def getEmbeddings(words):
	embeddings = []
	for word in words:
		word = word.lower()
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			# words not found in embedding index will be ignored.
			embeddings.append(embedding_vector)
	return embeddings
	
def hasEmbeddings(words):
	embeddings = []
	for word in words:
		word = word.lower()
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			# words not found in embedding index will be ignored.
			embeddings.append(word)
	return embeddings
	
def distance_matrix(pts):
    """Returns matrix of pairwise Euclidean distances. Vectorized numpy version."""
    return np.sum((pts[None,:] - pts[:, None])**2, -1)**0.5
	
def distance(a,b):
	counter_a = Counter(hasEmbeddings(a.strip()))
	counter_b = Counter(hasEmbeddings(b.strip()))
	uniq_words = list(set(counter_a.keys()).union(counter_b.keys()))
	
	bow_a = np.array([counter_a[w] for w in uniq_words], dtype=np.double)
	bow_b = np.array([counter_b[w] for w in uniq_words], dtype=np.double)
	
	bow_a /= bow_a.sum()
	bow_b /= bow_b.sum()
	embeddings = np.asarray(getEmbeddings(uniq_words))
	D = distance_matrix(embeddings)
	return emd(bow_a, bow_b, D)
	
def filter_story(story,question,limit):
	story = " ".join(story)
	question = " ".join(question)
	sentences = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', story)
	distances = []
	for sentence in sentences:
			distances.append(distance(sentence,question))
	return [sentences[i] for i in sorted(range(len(distances)), key=lambda x: distances[x])[-limit:]]

print('Found %s word vectors.' % len(embeddings_index))
no_lines = 500 if len(sys.argv)==1 else int(sys.argv[1]) 

with open(write_to, 'w') as w:
	with open(read_from, 'r') as r:
		i = 1
		for line in r:
			line = json.loads(line)
			story = line["string_sequence"]
			question = line["question_string_sequence"]
			answer = line["raw_answers"]
			sentences = filter_story(story,question,5)
			line = {}
			line["string_sequence"] = sentences
			line["question_string_sequence"] = question
			line["raw_answers"] = answer
			json.dump(line, w,ensure_ascii=False)
			if i > no_lines:
				break
			else:
				i = i+1
