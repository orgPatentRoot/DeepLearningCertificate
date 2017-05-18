# Structurally Contrained Recurrent Network (SCRN) Model
#
# This gives an implementation of the SCRN model given in Mikolov et al. 2015, arXiv:1412.7753 [cs.NE], 
# https://arxiv.org/abs/1412.7753 using Python and Tensorflow.
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
class scrn_graph(object):
    
    #
    def __init__(self, num_gpus, alpha, hidden_size, state_size, vocabulary_size, num_unfoldings, 
                 optimization_frequency, batch_size):
        
        #
        self._batch_size = batch_size
        self._num_gpus = num_gpus
        self._num_unfoldings = num_unfoldings
        self._optimization_frequency = optimization_frequency
        self._vocabulary_size = vocabulary_size
        
        #
        self._graph = tf.Graph()
        with self._graph.as_default():

            # Variable definitions:

            # Optimization variables
            self._learning_rate = tf.placeholder(tf.float32)
            
            # Context embedding tensor.
            B = tf.Variable(tf.truncated_normal([vocabulary_size, state_size], -0.1, 0.1))

            # Token embedding tensor.
            A = tf.Variable(tf.truncated_normal([vocabulary_size, hidden_size], -0.1, 0.1))
            
            #
            P = tf.Variable(tf.truncated_normal([state_size, hidden_size], -0.1, 0.1))

            # Recurrent weights tensor and bias.
            R = tf.Variable(tf.truncated_normal([hidden_size, hidden_size], -0.1, 0.1))

            # Output update tensor and bias.
            U = tf.Variable(tf.truncated_normal([hidden_size, vocabulary_size], -0.1, 0.1))
            V = tf.Variable(tf.truncated_normal([state_size, vocabulary_size], -0.1, 0.1))
            output_bias = tf.Variable(tf.zeros([vocabulary_size]))
            
            # Training data
            self._training_data = list()
            self._training_hidden_saved = list()
            self._training_state_saved = list()
            for _ in range(num_gpus):
                training_data_tmp = list()
                for _ in range(num_unfoldings + 1):
                    training_data_tmp.append(tf.placeholder(tf.float32, shape=[batch_size, vocabulary_size]))
                self._training_data.append(training_data_tmp)
                self._training_hidden_saved.append(tf.Variable(tf.zeros([self._batch_size, hidden_size]), trainable=False))
                self._training_state_saved.append(tf.Variable(tf.zeros([self._batch_size, state_size]), trainable=False))
            
            # Validation data
            self._validation_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size])
            validation_hidden_saved = tf.Variable(tf.zeros([1, hidden_size]))
            validation_state_saved = tf.Variable(tf.zeros([1, state_size]))
            
            #
            self._initialization = tf.global_variables_initializer()
            
            # Reset training state (this is a kludge to get moving on testing the rest of the code)
            if self._num_gpus == 1:
                self._reset_training_state = \
                    tf.group(self._training_hidden_saved[0].assign(tf.zeros([batch_size, hidden_size])),
                             self._training_state_saved[0].assign(tf.zeros([batch_size, state_size])))
            elif self._num_gpus == 2:
                self._reset_training_state = \
                    tf.group(self._training_hidden_saved[0].assign(tf.zeros([batch_size, hidden_size])),
                             self._training_state_saved[0].assign(tf.zeros([batch_size, state_size])),
                             self._training_hidden_saved[1].assign(tf.zeros([batch_size, hidden_size])),
                             self._training_state_saved[1].assign(tf.zeros([batch_size, state_size])))                          
            
            # Training:
            
            # Unfold SCRN
            
            #
            cost = list()
            gradients = list()
            labels = list()
            logits = list()
            training_hidden = list()
            training_input = list()
            training_label = list()
            training_labels = list()
            training_output = list()
            training_outputs = list()
            training_state = list()
            variables = list()
            for _ in range(self._num_gpus):
                cost.append([])
                gradients.append([])
                labels.append([])
                logits.append([])
                training_input.append([])
                training_label.append([])
                training_labels.append([])
                training_output.append([])
                training_outputs.append([])
                variables.append([])
            
            #
            optimizer = tf.train.GradientDescentOptimizer(self._learning_rate)
            for i in range(self._num_unfoldings // self._optimization_frequency):
                for tower in range(self._num_gpus):
                    cost[tower] = []
                    gradients[tower] = []
                    labels[tower] = []
                    logits[tower] = []
                    training_input[tower] = []
                    training_label[tower] = []
                    training_labels[tower] = []
                    training_output[tower] = []
                    training_outputs[tower] = []
                    variables[tower] = []
                training_hidden = self._training_hidden_saved
                training_state = self._training_state_saved
                for tower in range(self._num_gpus):
                    with tf.device('/gpu:%d' % tower):
                        for j in range(self._optimization_frequency):
                            training_input[tower] = self._training_data[tower][i*self._optimization_frequency + j]
                            training_label[tower] = self._training_data[tower][i*self._optimization_frequency + j + 1]
                            training_output[tower], training_hidden[tower], \
                                training_state[tower] = self._scrn_cell(training_input[tower], training_hidden[tower],
                                                                        training_state[tower], alpha, B, A, P, R, U,
                                                                        V, output_bias)
                            training_labels[tower].append(training_label[tower])
                            training_outputs[tower].append(training_output[tower])
                        logits[tower] = tf.concat(training_outputs[tower], 0)
                        labels[tower] = tf.concat(training_labels[tower], 0)
                        
                        # Replace with hierarchical softmax in the future
                        cost[tower] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels[tower],
                                                                                             logits=logits[tower]))
                        
                        gradients[tower], variables[tower] = zip(*optimizer.compute_gradients(cost[tower]))
                with tf.control_dependencies(self._set_training_saved(training_hidden, training_state)):
                    if i < self._num_unfoldings // self._optimization_frequency - 1:
                        optimizer.apply_gradients(zip(gradients[0], variables[0]))
            with tf.control_dependencies(self._set_training_saved(training_hidden, training_state)):
                logits = tf.concat(training_outputs[0], 0)
                labels = tf.concat(training_labels[0], 0)
                
                # Replace with hierarchical softmax in the future
                self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
                
            # Optimize parameters
            gradients, variables = zip(*optimizer.compute_gradients(self._cost))
            self._optimize = optimizer.apply_gradients(zip(gradients, variables))
                
            # Validation:
    
            # Reset validation state
            self._reset_validation_state = tf.group(validation_hidden_saved.assign(tf.zeros([1, hidden_size])),
                                                    validation_state_saved.assign(tf.zeros([1, state_size])))

            # Run SRN on validation data
            validation_output, validation_hidden, validation_state = self._scrn_cell(self._validation_input,
                                                                                     validation_hidden_saved,
                                                                                     validation_state_saved, alpha, B, A, P, R,
                                                                                     U, V, output_bias)
            with tf.control_dependencies([validation_hidden_saved.assign(validation_hidden), 
                                          validation_state_saved.assign(validation_state)]):
                logits = validation_output

                # Validation prediction, replace with hierarchical softmax in the future
                self._validation_prediction = tf.nn.softmax(logits)
                
    # SCRN cell definition:   .
    def _scrn_cell(self, x, h, s, alpha, B, A, P, R, U, V, output_bias):
        state_arg = (1 - alpha) * tf.matmul(x, B) + alpha * s
        state = state_arg
        hidden_arg = tf.matmul(state, P) + tf.matmul(x, A) + tf.matmul(h, R)
        hidden = tf.sigmoid(hidden_arg)
        output_arg = tf.matmul(hidden, U) + tf.matmul(state, V) 
        output = output_arg + output_bias
        return output, hidden, state
    
    #
    def _set_training_saved(self, training_hidden, training_state):
        for tower in range(self._num_gpus):
            self._training_hidden_saved[tower] = training_hidden[tower]
            self._training_state_saved[tower] = training_state[tower]
            
    # Optimization:
    def optimization(self, learning_rate, learning_decay, num_epochs, summary_frequency, training_text, validation_text):
        
        validation_size = len(validation_text)

        # Generate training batches
        print('Training Batch Generator:')
        training_batches = []
        for tower in range(self._num_gpus):
            print('     Tower: %d' % tower)
            training_batches.append(batch_generator(training_text[tower], self._batch_size,
                                                    self._num_unfoldings,self._vocabulary_size))
        
        # Generate validation batches
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
                for tower in range(self._num_gpus):
                    training_batches[tower].reset_token_idx()
                session.run(self._reset_training_state)
                for batch in range(training_batches[0].num_batches()):

                    # Get next training batch
                    training_batches_next = list()
                    for tower in range(self._num_gpus):
                        training_batches_next.append(training_batches[tower].next())
                    batch_ctr += 1

                    # Optimizationm
                    training_feed_dict[self._learning_rate] = learning_rate
                    for tower in range(self._num_gpus):
                        for i in range(self._num_unfoldings + 1):
                            training_feed_dict[self._training_data[tower][i]] = training_batches_next[tower][i]
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
                perplexity = float(2 ** (-validation_log_prob_sum / validation_size))
                print('Epoch: %d  Validation Set Perplexity: %.2f' % (epoch+1, perplexity))

                # Update learning rate
                if epoch > 0 and perplexity > perplexity_last_epoch:
                    learning_rate *= learning_decay
                perplexity_last_epoch = perplexity