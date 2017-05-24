# Implementation of "End to End Memory Networks" [https://arxiv.org/abs/1503.08895] for Wiki reading dataset
import os
import sys
import json
import random
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, add, dot, concatenate, multiply
from keras.layers import SimpleRNN

def vectorize(answer,word_idx):
	y = np.zeros(len(word_idx) + 1)
	for w in answer:
		y[word_idx[w]] = 1
	return y
	
def data_generator(batch_size,data, word_idx, story_length, query_length):
	while True:
		X = []
		Xq = []
		Y = []
		for i in range(int(batch_size)):
			# random choice
			(story, query, answer) = random.choice(data)
			x = [word_idx[w] for w in story]
			xq = [word_idx[w] for w in query]
			y = vectorize(answer,word_idx)
			X.append(x)
			Xq.append(xq)
			Y.append(y)
		yield ([pad_sequences(X, maxlen=story_length),pad_sequences(Xq, maxlen=query_length)], np.array(Y))

def build_vocab(data):
	vocab = set()
	for story, q, answer in data:
		vocab |= set(story + q + answer)
	return sorted(vocab)
		
filename = "../data/wiki/test/filtered-test-00000-of-00015.json"
data = []
no_lines = 100 if len(sys.argv)==1 else int(sys.argv[1]) 
steps = 10
with open(filename) as r:
	i = 1
	for line in r:
		line = json.loads(line)
		data.append((" ".join(line["string_sequence"]),line["question_string_sequence"],line["raw_answers"]))
		if i > no_lines:
			break
		else:
			i = i+1
vocab = build_vocab(data)
vocab_size = len(vocab) + 1
random.shuffle(data)
test_data = data[:int(len(data)/10)]
train_data = data[int(len(data)/10):]
story_length = max(map(len, (x for x, _, _ in train_data + test_data)))
query_length = max(map(len, (x for _, x, _ in train_data + test_data)))
print('Summary')
print('Vocab size:', vocab_size)
print('Training data:', len(train_data))
print('Test data:', len(test_data))
print('Story max length:', story_length)
print('Query max length:', query_length)
print('Data - [story, query, answer]:')
print('['," ".join(train_data[0][0]),", "," ".join(train_data[0][1]),", "," ".join(train_data[0][2]),"]")
exit()
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
# Embed the story sequence into m & c
story = Input((story_length,),dtype='int32')
encoder = Sequential()
encoder.add(Embedding(input_dim=vocab_size,output_dim=64))
m = encoder(story)
encoder = Sequential()
encoder.add(Embedding(input_dim=vocab_size,output_dim=query_length))
c = encoder(story)
# Embed the question sequence into u
question = Input((query_length,))
encoder = Sequential()
encoder.add(Embedding(input_dim=vocab_size,output_dim=64,input_length=query_length))
u = encoder(question)
# Compute  p = softmax(m * u)
p = dot([m, u], axes=(2, 2))
p = Activation('softmax')(p)
# Compute o = p * c  
o = multiply([p, c])
# Pass (o,u) through RNN for answer
answer = concatenate([Permute((2, 1))(o),u])
answer = SimpleRNN(64)(answer)
answer = Dense(vocab_size,kernel_initializer='random_normal')(answer)
answer = Activation('softmax')(answer)
# Model everything together
model = Model([story, question], answer)
model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
print(model.count_params())
# Train
model.fit_generator(data_generator(len(train_data)/steps,train_data,word_idx,story_length,query_length), 
			steps_per_epoch=steps, 
			nb_epoch=10,
			validation_data=data_generator(len(test_data)/steps,test_data,word_idx,story_length,query_length),
			validation_steps=steps)
