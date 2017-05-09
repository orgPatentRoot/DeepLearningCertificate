import sys
import json
import os
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
def build_vocab(X,Y):
	vocab = set()
	for story, question in X:
		vocab |= set(story + question)
	for answer in Y:
		vocab |= set(answer)
	return sorted(vocab)
	
def write_data(filename,X,Y):
	with open(filename, 'w') as w:
		for index, elem in enumerate(X):
			line = {}
			line["string_sequence"] = elem[0]
			line["question_string_sequence"] = elem[1]
			line["raw_answers"] = Y[index]
			json.dump(line, w,ensure_ascii=False)

data_location = "../data/wiki/sanitized/data/"
test_location = "../data/wiki/sanitized/test/"
train_location = "../data/wiki/sanitized/train/"
validation_location = "../data/wiki/sanitized/validation/"
metadata_location = "../data/wiki/sanitized/metadata/"
vocab_location = "../data/wiki/sanitized/vocab/"
metadata = {}
X = []
Y = []
filename = "0" if len(sys.argv)==1 else sys.argv[1]
fullname = data_location + filename + ".json"
with open(fullname, 'r') as r:
	for line in r:
		line = json.loads(line)
		X.append((line["string_sequence"],line["question_string_sequence"]))
		Y.append(line["raw_answers"])
vocab = build_vocab(X,Y)
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
metadata["vocab_size"] = len(vocab) + 1
metadata["story_max_length"] = max(map(len, (x for x, _ in X)))
metadata["query_max_length"] = max(map(len, (x for _, x in X)))
metadata["answer_max_length"] = max(map(len, (x for x in Y)))
fullname = vocab_location + filename + ".p"
pickle.dump( word_idx, open( fullname, "wb" ) )
X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)
X_valid, X_test, Y_valid, Y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
metadata["train_data_size"] = len(Y_train)
metadata["valid_data_size"] = len(Y_valid)
metadata["test_data_size"] = len(Y_test)
fullname = metadata_location + filename + ".json"
with open(fullname, 'w') as outfile:
    json.dump(metadata, outfile)
fullname = train_location + filename + ".json"
write_data(fullname,X_train,Y_train)
fullname = validation_location + filename + ".json"
write_data(fullname,X_valid,Y_valid)
fullname = test_location + filename + ".json"
write_data(fullname,X_test,Y_test)


