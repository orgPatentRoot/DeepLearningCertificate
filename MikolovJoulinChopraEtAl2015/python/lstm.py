# Structurally Contrained Recurrent Network (SCRN) Model
#
# This gives an implementation of the LSTM model for comparison with the SCRN model given in Mikolov et al. 2015,
# arXiv:1412.7753 [cs.NE], https://arxiv.org/abs/1412.7753 using Python and Tensorflow.
#
# This code fails to implement hierarchical softmax at this time as Tensorflow does not appear to include an
# implementation.  Hierarchical softmax can be included at a future date when hierarchical softmax is available 
# for Tensorflow or by modifying the code to run in Keras which appears to have an implementation of hierarchical
# softmax.
#
# Stuart Hagler, 2017

# Imports
import math
import numpy as np
import tensorflow as tf

# Local imports
from batch_generator import batch_generator
from log_prob import log_prob

# Tensorflow graph
class lstm_graph(object):
    
    #
    def __init__(self, hidden_size, vocabulary_size, num_unfoldings, batch_size):
        
        #
        self._batch_size = batch_size
        self._num_unfoldings = num_unfoldings
        self._vocabulary_size = vocabulary_size
        
        #
        self._graph = tf.Graph()
        with self._graph.as_default():

            # Variable definitions:
            
            # Optimization variables
            self._clip_norm = tf.placeholder(tf.float32)
            self._learning_rate = tf.placeholder(tf.float32)
            self._optimization_frequency = tf.placeholder(tf.int32)

            # Forget gate input and output tensor and bias.
            Wf = tf.Variable(tf.truncated_normal([vocabulary_size, hidden_size], -0.1, 0.1))
            Uf = tf.Variable(tf.truncated_normal([hidden_size, hidden_size], -0.1, 0.1))
            forget_bias = tf.Variable(tf.zeros([1, hidden_size]))

            # Input gate input and output tensor and bias.
            Wi = tf.Variable(tf.truncated_normal([vocabulary_size, hidden_size], -0.1, 0.1))
            Ui = tf.Variable(tf.truncated_normal([hidden_size, hidden_size], -0.1, 0.1))
            input_bias = tf.Variable(tf.zeros([1, hidden_size]))

            # Output gate input and output tensor and bias.
            Wo = tf.Variable(tf.truncated_normal([vocabulary_size, hidden_size], -0.1, 0.1))
            Uo = tf.Variable(tf.truncated_normal([hidden_size, hidden_size], -0.1, 0.1))
            output_bias = tf.Variable(tf.zeros([1, hidden_size]))

            # Cell state update input and output tensor and bias.
            Wc = tf.Variable(tf.truncated_normal([vocabulary_size, hidden_size], -0.1, 0.1))
            Uc = tf.Variable(tf.truncated_normal([hidden_size, hidden_size], -0.1, 0.1))
            update_bias = tf.Variable(tf.zeros([1, hidden_size]))

            # Softmax weight tensor and bias.
            W = tf.Variable(tf.truncated_normal([hidden_size, vocabulary_size], -0.1, 0.1))
            W_bias = tf.Variable(tf.zeros([vocabulary_size]))
            
            # Training data
            self._training_data = list()
            for _ in range(num_unfoldings + 1):
                self._training_data.append(tf.placeholder(tf.float32, shape=[batch_size, vocabulary_size]))
            training_output_saved = tf.Variable(tf.zeros([self._batch_size, hidden_size]), trainable=False)
            training_state_saved = tf.Variable(tf.zeros([self._batch_size, hidden_size]), trainable=False)
            
            # Validation data
            self._validation_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size])
            validation_output_saved = tf.Variable(tf.zeros([1, hidden_size]))
            validation_state_saved = tf.Variable(tf.zeros([1, hidden_size]))
            
            #
            self._initialization = tf.global_variables_initializer()
            
            # Reset training state
            self._reset_training_state = tf.group(training_output_saved.assign(tf.zeros([batch_size, hidden_size])),
                                                  training_state_saved.assign(tf.zeros([batch_size, hidden_size])))
            
            # Training:
            
            # Unfold LSTM
            training_output = training_output_saved
            training_state = training_state_saved
            training_labels = []
            training_outputs = []
            optimize_ctr = 0
            for i in range(self._num_unfoldings):
                training_input = self._training_data[i]
                training_label = self._training_data[i+1]
                training_output, training_state = self._lstm_cell(training_input, training_output, training_state, 
                    Wf, Uf, forget_bias, Wi, Ui, input_bias, Wo, Uo, output_bias, Wc, Uc, update_bias)
                training_labels.append(training_label)
                training_outputs.append(training_output)
                optimize_ctr += 1
                if optimize_ctr < self._num_unfoldings and optimize_ctr % self._optimization_frequency == 0:
                    logits = tf.nn.xw_plus_b(tf.concat(training_outputs, 0), W, W_bias)
                    labels = tf.concat(training_labels, 0)
                    self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
                    gradients, variables = zip(*optimizer.compute_gradients(self._cost))
                    gradients, _ = tf.clip_by_global_norm(gradients, self._clip_norm)
                    optimizer.apply_gradients(zip(gradients, variables))
            with tf.control_dependencies([training_output_saved.assign(training_output), 
                                          training_state_saved.assign(training_state)]):
                logits = tf.nn.xw_plus_b(tf.concat(training_outputs, 0), W, W_bias)
                labels = tf.concat(training_labels, 0)
                
                # Replace with hierarchical softmax in the future
                self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
                
            # Optimizer.
            optimizer = tf.train.GradientDescentOptimizer(self._learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self._cost))
            gradients, _ = tf.clip_by_global_norm(gradients, self._clip_norm)
    
            # Optimize parameters
            self._optimize = optimizer.apply_gradients(zip(gradients, variables))
                
            # Validation:
    
            # Reset validation state
            self._reset_validation_state = tf.group(validation_output_saved.assign(tf.zeros([1, hidden_size])),
                                                    validation_state_saved.assign(tf.zeros([1, hidden_size])))

            # Run LSTM on validation data
            validation_output, validation_state = self._lstm_cell(self._validation_input, validation_output_saved,
                validation_state_saved, Wf, Uf, forget_bias, Wi, Ui, input_bias, Wo, Uo, output_bias, Wc, Uc, update_bias)
            with tf.control_dependencies([validation_output_saved.assign(validation_output), 
                                          validation_state_saved.assign(validation_state)]):
                logits = tf.nn.xw_plus_b(validation_output, W, W_bias)

                # Validation prediction, replace with hierarchical softmax in the future
                self._validation_prediction = tf.nn.softmax(logits)
                
    # LSTM cell definition:   .
    def _lstm_cell(self, x, h, c, Wf, Uf, forget_bias, Wi, Ui, input_bias, Wo, Uo, output_bias, Wc, Uc, update_bias):
        forget_arg = tf.matmul(x, Wf) + tf.matmul(h, Uf)
        forget_gate = tf.sigmoid(forget_arg + forget_bias)
        input_arg = tf.matmul(x, Wi) + tf.matmul(h, Ui)
        input_gate = tf.sigmoid(input_arg + input_bias)
        output_arg = tf.matmul(x, Wo) + tf.matmul(h, Uo)
        output_gate = tf.sigmoid(output_arg + output_bias)
        update_arg = tf.matmul(x, Wc) + tf.matmul(h, Uc)
        state = forget_gate * c + input_gate * tf.tanh(update_arg + update_bias)
        output = output_gate * tf.tanh(state)
        return output, state
            
    # Optimization:
    def optimization(self, learning_rate, learning_decay, optimization_frequency, clip_norm, num_epochs, summary_frequency,
                     training_text, validation_text):
        
        training_size = len(training_text)
        validation_size = len(validation_text)
        
        print('Training Batch Generator:')
        training_batches = batch_generator(training_text, self._batch_size, self._num_unfoldings, self._vocabulary_size)
        
        print('Validation Batch Generator:')
        validation_batches = batch_generator(validation_text, 1, 1, self._vocabulary_size)
        
        batch_ctr = 0
        epoch_ctr = 0
        training_feed_dict = dict()
        validation_feed_dict = dict()
        
        with tf.Session(graph=self._graph) as session:
        
            session.run(self._initialization)
            print('Initialized')

            # Iterate over fixed number of training epochs
            for epoch in range(num_epochs):

                # Display the learning rate for this epoch
                print('Epoch: %d  Learning Rate: %.2f' % (epoch+1, learning_rate))

                # Optimization Step:

                # Iterate over training batches
                training_batches.reset_token_idx()
                session.run(self._reset_training_state)
                for batch in range(training_batches.num_batches()):

                    # Get next training batch
                    train_batches_next = training_batches.next()
                    batch_ctr += 1

                    # Optimization
                    training_feed_dict[self._clip_norm] = clip_norm
                    training_feed_dict[self._learning_rate] = learning_rate
                    training_feed_dict[self._optimization_frequency] = optimization_frequency
                    for i in range(self._num_unfoldings + 1):
                        training_feed_dict[self._training_data[i]] = train_batches_next[i]
                    session.run(self._optimize, feed_dict=training_feed_dict)

                    # Summarize current performance
                    if (batch+1) % summary_frequency == 0:
                        cst = session.run(self._cost, feed_dict=training_feed_dict)
                        print('     Total Batches: %d  Current Batch: %d  Cost: %.2f' % 
                              (batch_ctr, batch+1, cst))
                        
                # Validation Step:
        
                # Iterate over validation batches
                validation_batches.reset_token_idx()
                session.run(self._reset_validation_state)
                validation_log_prob_sum = 0
                for i in range(validation_size):
                    
                    # Get next validation batch
                    validation_batch_next = validation_batches.next()
                    
                    # Validation
                    validation_feed_dict[self._validation_input] = validation_batch_next[0]
                    validation_batch_next_label = validation_batch_next[1]
                    validation_prediction = session.run(self._validation_prediction, feed_dict=validation_feed_dict)
                    
                    # Summarize current performance
                    validation_log_prob_sum = validation_log_prob_sum + log_prob(validation_prediction, 
                                                                                 validation_batch_next_label)
                
                #
                perplexity = float(2 ** (-validation_log_prob / validation_size))
                print('Epoch: %d  Validation Set Perplexity: %.2f' % (epoch+1, perplexity))

                # Update learning rate
                if epoch > 0 and perplexity > perplexity_last_epoch:
                    learning_rate *= learning_decay
                perplexity_last_epoch = perplexity