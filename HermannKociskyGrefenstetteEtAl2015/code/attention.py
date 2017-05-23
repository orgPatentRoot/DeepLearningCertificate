# Implementation of "Attentive Reader Model" [https://arxiv.org/abs/1506.03340] for Wiki reading dataset
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
from keras.layers import Input, Activation, Dense, Permute, add, dot, concatenate, multiply, Dropout, RepeatVector, Merge, Flatten, Reshape
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers.wrappers import TimeDistributed
from keras import backend as K

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
embedding_dim = 32
lstm_dim = 32
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

# Embed the story sequence
story = Input((story_length,),dtype='int32')
encoder = Sequential()
encoder.add(Embedding(input_dim=vocab_size,output_dim=embedding_dim))
encoder.add(Dropout(0.3))
encoded_story = encoder(story)
# Use the embedded story to lstm forward
story_lstm_forward = LSTM(lstm_dim,return_sequences = True)(encoded_story)
story_lstm_forward = Dropout(0.3)(story_lstm_forward)
# Use the embedded story to lstm backward
story_lstm_backward = LSTM(lstm_dim,return_sequences = True,go_backwards=True)(encoded_story)
story_lstm_backward = Dropout(0.3)(story_lstm_backward)
# Concactenate both forward and backward
yd = concatenate([story_lstm_forward, story_lstm_backward])
story_dense = TimeDistributed(Dense(2*lstm_dim))(yd)

# Embed the question sequence
question = Input((query_length,))
encoder = Sequential()
encoder.add(Embedding(input_dim=vocab_size,output_dim=embedding_dim,input_length=query_length))
encoder.add(Dropout(0.3))
encoded_question = encoder(question)
# Use the embedded question to lstm forward
query_lstm_forward = LSTM(lstm_dim)(encoded_question)
query_lstm_forward = Dropout(0.3)(query_lstm_forward)
# Use the embedded question to lstm backward
query_lstm_backward = LSTM(lstm_dim,go_backwards=True)(encoded_question)
query_lstm_backward = Dropout(0.3)(query_lstm_backward)
# Concactenate both forward and backward
u = concatenate([query_lstm_forward, query_lstm_backward])
# Stretch it to match story length dimension
u_rpt = RepeatVector(story_length)(u)
query_dense = TimeDistributed(Dense(2*lstm_dim))(u_rpt)
# Sum up story and query
story_query_sum = add([story_dense, query_dense])
m = Activation('tanh')(story_query_sum)
w_m = TimeDistributed(Dense(1))(m)
w_m_flat = Flatten()(w_m)
s = Activation('softmax')(w_m_flat)
s = Reshape((story_length,1))(s)
r = Flatten()(dot([s, yd],axes=(1,1)))
g_r = Dense(embedding_dim)(r)
g_u = Dense(embedding_dim)(u)
g_r_plus_g_u = add([g_r, g_u])
g_d_q = Activation('tanh')(g_r_plus_g_u)
answer = Dense(vocab_size, activation='softmax')(g_d_q)
# Model everything together
model = Model([story, question], answer)
model.compile(optimizer='RMSprop', loss='categorical_crossentropy',metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001)
print(model.count_params())
check_point_location = "../data/wiki/model/attention/" + filename + ".hdf5"
# CheckPoints
checkpointer = ModelCheckpoint(filepath=check_point_location, verbose=1, save_best_only=False)
# Train
history = model.fit_generator(data_generator(train_batch_size,train_data_file_name,train_data_size,word_idx,story_length,query_length), 
		steps_per_epoch=train_steps, 
		nb_epoch=64,
		validation_data=data_generator(valid_batch_size,valid_data_file_name,valid_data_size,word_idx,story_length,query_length),
		validation_steps=valid_steps,
		callbacks=[checkpointer,reduce_lr])
print(model.summary())
result = model.evaluate_generator(data_generator(test_batch_size,test_data_file_name,test_data_size,word_idx,story_length,query_length), 
		steps=test_steps, 
		max_q_size=10, 
		workers=1, 
		pickle_safe=False)
print(result)
# save history
location = "../data/wiki/history/attention"
fullname = location + filename + ".json"
with open(fullname, 'w') as f:
	json.dump(history.history, f)
