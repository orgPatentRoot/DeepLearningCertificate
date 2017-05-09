# -*- coding: utf-8 -*-
"""
===================================
Demo of DBSCAN clustering algorithm
===================================

Finds core samples of high density and expands clusters from them.

"""
print(__doc__)

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

EMBED_DIR = '/home/ubuntu/data/embed'
EMBED_NAME = 'wiki.en.vec'
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

print('Found %s word vectors.' % len(embeddings_index))

def getAverageEmbeddings(words):
	featureVec = np.zeros((300,),dtype="float64")
	embeddings = []
	count = 0
	for word in words:
		word = word.lower()
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			# words not found in embedding index will be ignored.
			featureVec = np.add(featureVec,embedding_vector)
			count = count + 1
	if count != 0:
		featureVec = np.divide(featureVec,count)
	return featureVec

##############################################################################
# Generate sample data
filename = "../data/wiki/filtered/filtered-test-00000-of-00015.json"
X = []
with open(filename) as r:
	for line in r:
		line = json.loads(line)
		X.append(np.stack([getAverageEmbeddings(line["question_string_sequence"]),getAverageEmbeddings(line["raw_answers"])]))

##############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
