# Error Analyze the model with examples
import os
import sys
import json
import random
import numpy as np
import pickle
import linecache
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, add, dot, concatenate, multiply, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def random_sample(file_name,file_size,k):
	data = []
	line_numbers = sorted(random.sample(range(1,file_size), k))
	for line_number in line_numbers:
		line = linecache.getline(file_name, line_number)
		line = json.loads(line.strip())
		data.append((line["string_sequence"],line["question_string_sequence"],line["raw_answers"]))
	linecache.clearcache()
	return data

def vectorize(answer,word_idx):
	y = np.zeros(len(word_idx) + 1)
	for w in answer:
		y[word_idx[w]] = 1
	return y

def de_vectorize(y,idx_word,answer_length):
	answer = []
	top_indices = y.argsort()[-answer_length:][::-1]
	for index in top_indices:
		if index in idx_word:
			answer.append(idx_word[index])
	return answer
	
def vectorize_data(data,word_idx, story_length, query_length):
	X = []
	Xq = []
	Y = []
	for story, query, answer in data:
		x = [word_idx[w] for w in story]
		xq = [word_idx[w] for w in query]
		y = vectorize(answer,word_idx)
		X.append(x)
		Xq.append(xq)
		Y.append(y)
	return ([pad_sequences(X, maxlen=story_length),pad_sequences(Xq, maxlen=query_length)], np.array(Y))
	
	
def data_generator(batch_size,file_name,file_size,word_idx, story_length, query_length):
	while True:
		X = []
		Xq = []
		Y = []
		data = random_sample(file_name, file_size,batch_size)
		for story, query, answer in data:
			x = [word_idx[w] for w in story]
			xq = [word_idx[w] for w in query]
			y = vectorize(answer,word_idx)
			X.append(x)
			Xq.append(xq)
			Y.append(y)
		yield ([pad_sequences(X, maxlen=story_length),pad_sequences(Xq, maxlen=query_length)], np.array(Y))

test_location = "../data/wiki/sanitized/test/"
train_location = "../data/wiki/sanitized/train/"
validation_location = "../data/wiki/sanitized/validation/"
metadata_location = "../data/wiki/sanitized/metadata/"
vocab_location = "../data/wiki/sanitized/vocab/"
metadata = {}
filename = "0" if len(sys.argv)==1 else sys.argv[1]
train_data_file_name = train_location + filename + ".json"
valid_data_file_name = validation_location + filename + ".json"
test_data_file_name = test_location + filename + ".json"
fullname = vocab_location + filename + ".p"
word_idx = pickle.load( open( fullname, "rb" ) )
idx_word = {v: k for k, v in word_idx.items()}
train_batch_size = 100
valid_batch_size = 50
test_batch_size = 50
embedding_dim = 300
fullname = metadata_location + filename + ".json"
with open(fullname) as data_file:    
    metadata = json.load(data_file)
vocab_size = int(metadata["vocab_size"])
train_data_size = int(metadata["train_data_size"])
valid_data_size = int(metadata["valid_data_size"])
test_data_size = int(metadata["test_data_size"])
vocab_size = int(metadata["vocab_size"])
train_steps = int(train_data_size/train_batch_size) 
valid_steps = int(valid_data_size/valid_batch_size)
test_steps = int(test_data_size/test_batch_size)
story_length = int(metadata["story_max_length"])
query_length = int(metadata["query_max_length"])
answer_length = int(metadata["answer_max_length"])
print('Summary')
print('Vocab size:', vocab_size)
print('Training data:', train_data_size)
print('Validation data:', valid_data_size)
print('Story max length:', story_length)
print('Query max length:', query_length)
check_point_location = "../data/wiki/model/memory/" + filename + ".hdf5"
model = load_model(check_point_location)
result = model.evaluate_generator(data_generator(test_batch_size,test_data_file_name,test_data_size,word_idx,story_length,query_length), 
		steps=test_steps, 
		max_q_size=10, 
		workers=1, 
		pickle_safe=False)
data = random_sample(test_data_file_name,test_data_size,10)
(X,Y) = vectorize_data(data,word_idx, story_length, query_length)
predictions = model.predict(X, batch_size=test_batch_size, verbose=1)
for index,prediction in enumerate(predictions):
	answer = de_vectorize(prediction,idx_word,answer_length)
	if len(answer) > 0:
		print(data[index][0])
		print(data[index][1])
		print(answer)
		print(data[index][2])
print(result)
