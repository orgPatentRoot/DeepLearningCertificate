# Structurally Contrained Recurrent Network (SCRN) Model
#
# This gives an implementation of the SRN model for comparison with the SCRN model given in Mikolov et al. 2015,
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
class srn_graph(object):
    
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

            # Token embedding tensor.
            A = tf.Variable(tf.truncated_normal([vocabulary_size, hidden_size], -0.1, 0.1))

            # Recurrent weights tensor and bias.
            R = tf.Variable(tf.truncated_normal([hidden_size, hidden_size], -0.1, 0.1))
            hidden_bias = tf.Variable(tf.zeros([hidden_size]))

            # Output update tensor and bias.
            U = tf.Variable(tf.truncated_normal([hidden_size, vocabulary_size], -0.1, 0.1))
            output_bias = tf.Variable(tf.zeros([vocabulary_size]))
            
            # Training data
            self._training_data = list()
            for _ in range(num_unfoldings + 1):
                self._training_data.append(tf.placeholder(tf.float32, shape=[batch_size, vocabulary_size]))
            training_hidden_saved = tf.Variable(tf.zeros([batch_size, hidden_size]), trainable=False)
            
            # Validation data
            self._validation_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size])
            validation_hidden_saved = tf.Variable(tf.zeros([1, hidden_size]))
            
            #
            self._initialization = tf.global_variables_initializer()
            
            # Reset training state
            self._reset_training_state = tf.group(training_hidden_saved.assign(tf.zeros([batch_size, hidden_size])))
            
            # Training:
            
            # Unfold SRN
            training_hidden = training_hidden_saved
            training_labels = []
            training_outputs = []
            optimize_ctr = 0
            for i in range(self._num_unfoldings):
                training_input = self._training_data[i]
                training_label = self._training_data[i+1]
                training_output, training_hidden = self._srn_cell(training_input, training_hidden, A, R, hidden_bias, U,
                                                                  output_bias)
                training_labels.append(training_label)
                training_outputs.append(training_output)
                optimize_ctr += 1
                if optimize_ctr < self._num_unfoldings and optimize_ctr % self._optimization_frequency == 0:
                    logits = tf.concat(training_outputs, 0)
                    labels = tf.concat(training_labels, 0)
                    self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
                    gradients, variables = zip(*optimizer.compute_gradients(self._cost))
                    gradients, _ = tf.clip_by_global_norm(gradients, self._clip_norm)
                    optimizer.apply_gradients(zip(gradients, variables))
            with tf.control_dependencies([training_hidden_saved.assign(training_hidden)]):
                logits = tf.concat(training_outputs, 0)
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
            self._reset_validation_state = tf.group(validation_hidden_saved.assign(tf.zeros([1, hidden_size])))

            # Run SRN on validation data
            validation_output, validation_hidden = self._srn_cell(self._validation_input, validation_hidden_saved, A, R,
                                                                  hidden_bias, U, output_bias)
            with tf.control_dependencies([validation_hidden_saved.assign(validation_hidden)]):
                logits = validation_output

                # Validation prediction, replace with hierarchical softmax in the future
                self._validation_prediction = tf.nn.softmax(logits)
                
    # SRN cell definition:   .
    def _srn_cell(self, x, h, A, R, hidden_bias, U, output_bias):
        hidden_arg = tf.matmul(x, A) + tf.matmul(h, R)
        hidden = tf.sigmoid(hidden_arg + hidden_bias)
        output_arg = tf.matmul(hidden, U)
        output = output_arg + output_bias
        return output, hidden
            
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
                validation_logprob = 0
                for i in range(validation_size):
                    
                    # Get next validation batch
                    validation_batches_next = validation_batches.next()
                    
                    # Validation
                    validation_feed_dict[self._validation_input] = validation_batches_next[0]
                    validation_batches_next_labels = validation_batches_next[1]
                    validation_predictions = session.run(self._validation_prediction, feed_dict=validation_feed_dict)
                    
                    # Summarize current performance
                    validation_logprob = validation_logprob + log_prob(validation_predictions, 
                                                                       validation_batches_next_labels)
                    
                # 
                perplexity = float(np.exp(validation_logprob / validation_size))
                print('Epoch: %d  Validation Set Perplexity: %.2f' % (epoch+1, perplexity))

                # Update learning rate
                if epoch > 0 and perplexity > perplexity_last_epoch:
                    learning_rate *= learning_decay
                perplexity_last_epoch = perplexity