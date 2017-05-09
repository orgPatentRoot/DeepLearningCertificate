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
from keras.layers import Input, Activation, Dense, Permute, add, dot, concatenate, multiply, Dropout, Embedding, RepeatVector
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from metrics import precision,recall,fmeasure

def random_sample(file_name, file_size,k):
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
embedding_dim = 64
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
story_length = int(metadata["story_max_length"])
query_length = int(metadata["query_max_length"])
print('Summary')
print('Vocab size:', vocab_size)
print('Training data:', train_data_size)
print('Validation data:', valid_data_size)
print('Story max length:', story_length)
print('Query max length:', query_length)

# embed story into encoded story
story = Input(shape=(story_length,), dtype='int32')
encoded_story = Embedding(vocab_size, embedding_dim)(story)
encoded_story = Dropout(0.3)(encoded_story)
# embed question into encoded question
question = Input(shape=(query_length,), dtype='int32')
encoded_question = Embedding(vocab_size, embedding_dim)(question)
encoded_question = Dropout(0.3)(encoded_question)
encoded_question = LSTM(embedding_dim)(encoded_question)
encoded_question = RepeatVector(story_length)(encoded_question)
# merge both into RNN
merged = add([encoded_story, encoded_question])
merged = LSTM(embedding_dim)(merged)
merged = Dropout(0.3)(merged)
answer = Dense(vocab_size, activation='softmax')(merged)

model = Model([story, question], answer)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001)
print(model.count_params())
check_point_location = "../data/wiki/model/baseline/" + filename + ".hdf5"
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
		workers=4, 
		pickle_safe=False)
print(result)
# save history
location = "../data/wiki/history/baseline/"
fullname = location + filename + ".json"
with open(fullname, 'w') as f:
	json.dump(history.history, f)
