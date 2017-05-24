# Implementation of "End to End Memory Networks" [https://arxiv.org/abs/1503.08895] for Wiki reading dataset
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

print('Summary')
print('Vocab size:', vocab_size)
print('Training data:', train_data_size)
print('Validation data:', valid_data_size)
print('Story max length:', story_length)
print('Query max length:', query_length)
# Embed the story sequence into m & c
story = Input((story_length,),dtype='int32')
print("story:",story.shape)
encoder = Sequential()
encoder.add(Embedding(input_dim=vocab_size,output_dim=embedding_dim))
encoder.add(Dropout(0.3))
m = encoder(story)
print("m:",m.shape)
encoder = Sequential()
encoder.add(Embedding(input_dim=vocab_size,output_dim=query_length))
encoder.add(Dropout(0.3))
c = encoder(story)
print("c:",c.shape)
# Embed the question sequence into u
question = Input((query_length,))
print("question:",question.shape)
encoder = Sequential()
encoder.add(Embedding(input_dim=vocab_size,output_dim=embedding_dim,input_length=query_length))
encoder.add(Dropout(0.3))
u = encoder(question)
print("u:",u.shape)
# Compute  p = softmax(m * u)
p = dot([m, u], axes=(2, 2))
p = Activation('softmax')(p)
print("p:",p.shape)
# Compute o = p * c  
o = multiply([p, c])
print("o:",o.shape)
# Pass (o,u) through RNN for answer
answer = concatenate([Permute((2, 1))(o),u])
answer = LSTM(32)(answer)
answer = Dropout(0.3)(answer)
answer = Dense(vocab_size,kernel_initializer='random_normal')(answer)
answer = Activation('softmax')(answer)
print("answer:",answer.shape)
# Model everything together
model = Model([story, question], answer)
model.compile(optimizer='RMSprop', loss='categorical_crossentropy',metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001)
print(model.count_params())
check_point_location = "../data/wiki/model/memory/" + filename + ".hdf5"
# CheckPoints
checkpointer = ModelCheckpoint(filepath=check_point_location, verbose=1, save_best_only=False)
# Train
history = model.fit_generator(data_generator(train_batch_size,train_data_file_name,train_data_size,word_idx,story_length,query_length), 
		steps_per_epoch=train_steps, 
		nb_epoch=64,
		validation_data=data_generator(valid_batch_size,valid_data_file_name,valid_data_size,word_idx,story_length,query_length),
		validation_steps=valid_steps,
		callbacks=[checkpointer,reduce_lr])
result = model.evaluate_generator(data_generator(test_batch_size,test_data_file_name,test_data_size,word_idx,story_length,query_length), 
		steps=test_steps, 
		max_q_size=10, 
		workers=1, 
		pickle_safe=False)
print(result)
# save history
location = "../data/wiki/history/memory"
fullname = location + filename + ".json"
with open(fullname, 'w') as f:
	json.dump(history.history, f)
