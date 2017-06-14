# Delta Recurrent Neural Network (Delta-RNN) Framework
#
# This gives an implementation of the Delta-RNN framework given in Ororbia et al. 2017, arXiv:1703.08864 [cs.CL], 
# https://arxiv.org/abs/1703.08864 using Python and Tensorflow.
#
# This code implements a variety of RNN models using the Delta-RNN Framework
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
class delta_rnn_graph(object):
    
    # Graph constructor
    def __init__(self, cell_flag, num_gpus, alpha, c_size, h_size, vocabulary_size, num_training_unfoldings,
                 num_validation_unfoldings, batch_size, optimization_frequency, clip_norm, momentum):
        
        # Input hyperparameters
        self._alpha = alpha
        self._batch_size = batch_size
        self._c_size = c_size
        self._cell_flag = cell_flag
        self._clip_norm = clip_norm
        self._h_size = h_size
        self._momentum = momentum
        self._num_gpus = num_gpus
        self._num_training_unfoldings = num_training_unfoldings
        self._num_validation_unfoldings = num_validation_unfoldings
        self._optimization_frequency = optimization_frequency
        self._vocabulary_size = vocabulary_size
        
        # Derived hyperparameters
        self._num_towers = self._num_gpus
        
        # Graph definition
        self._graph = tf.Graph()
        with self._graph.as_default():

            # Cell parameter definitions
            self._setup_cell_parameters_wrapper()
            
            # Training data
            self._training_data = []
            self._training_c_saved = []
            self._training_h_saved = []
            for _ in range(self._num_towers):
                training_data_tmp = []
                for _ in range(num_training_unfoldings + 1):
                    training_data_tmp.append(tf.placeholder(tf.float32, shape=[self._batch_size, self._vocabulary_size]))
                self._training_data.append(training_data_tmp)
                self._training_c_saved.append(tf.Variable(tf.zeros([self._batch_size, self._c_size]),
                                                          trainable=False))
                self._training_h_saved.append(tf.Variable(tf.zeros([self._batch_size, self._h_size]),
                                                          trainable=False))
                
            # Validation data
            self._validation_input = []
            self._validation_c_saved = []
            self._validation_h_saved = []
            for _ in range(self._num_towers):
                validation_input_tmp = []
                for _ in range(num_validation_unfoldings):
                    validation_input_tmp.append(tf.placeholder(tf.float32, shape=[1, self._vocabulary_size]))
                self._validation_input.append(validation_input_tmp)
                self._validation_c_saved.append(tf.Variable(tf.zeros([1, self._c_size]), trainable=False))
                self._validation_h_saved.append(tf.Variable(tf.zeros([1, self._h_size]), trainable=False))
                
            # Optimizer hyperparameters
            self._learning_rate = tf.placeholder(tf.float32)
                
            # Optimizer
            self._optimizer = tf.train.MomentumOptimizer(self._learning_rate, self._momentum)
                    
            # Training:
            
            # Reset training state
            self._reset_training_state = \
                [ tf.group(self._training_c_saved[tower].assign(tf.zeros([self._batch_size, self._c_size])),
                           self._training_h_saved[tower].assign(tf.zeros([self._batch_size, self._h_size]))) \
                  for tower in range(self._num_towers) ]
            
            # Train cell on training data
            for i in range(self._num_training_unfoldings // self._optimization_frequency):
                training_labels = []
                training_outputs = []
                for tower in range(self._num_towers):
                    training_labels.append([])
                    training_outputs.append([])
                for tower in range(self._num_towers):
                    training_outputs[tower], training_labels[tower] = \
                        self._training_tower(i, tower, tower)
                all_training_outputs = []
                all_training_labels = []
                for tower in range(self._num_towers):
                    all_training_outputs += training_outputs[tower]
                    all_training_labels += training_labels[tower]
                logits = tf.concat(all_training_outputs, 0)
                labels = tf.concat(all_training_labels, 0)

                # Replace with hierarchical softmax in the future
                self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

                gradients, variables = zip(*self._optimizer.compute_gradients(self._cost))
                gradients, _ = tf.clip_by_global_norm(gradients, self._clip_norm)
                self._optimize = self._optimizer.apply_gradients(zip(gradients, variables))
                
            # Initialization:
            
            self._initialization = tf.global_variables_initializer()
                
            # Validation:
    
            # Reset validation state
            self._reset_validation_state = \
                [ tf.group(self._validation_c_saved[tower].assign(tf.zeros([1, self._c_size])),
                           self._validation_h_saved[tower].assign(tf.zeros([1, self._h_size]))) \
                  for tower in range(self._num_towers) ]

            # Run cell on validation data
            validation_outputs = []
            for tower in range(self._num_towers):
                validation_outputs.append([])
            for tower in range(self._num_towers):
                validation_outputs[tower] = self._validation_tower(tower, tower)
            logits = validation_outputs

            # Validation prediction, replace with hierarchical softmax in the future
            self._validation_prediction = tf.nn.softmax(logits)
            
    # RNN cells:
    
    # GRU cell definition
    def _gru_cell(self, x, c, h):
        z = tf.sigmoid(tf.matmul(x, self._Wz) + tf.matmul(h, self._Vz) + self._bz)
        r = tf.sigmoid(tf.matmul(x, self._Wr) + tf.matmul(h, self._Vr) + self._br)
        c = tf.matmul(x, self._Wh) + tf.matmul(r*h, self._Vh) + self._bh
        h = z*h + (1-z)*tf.tanh(c)
        o = tf.nn.xw_plus_b(h, self._W, self._b)
        return o, c, h
    
    # Peephole LSTM cell definition
    def _peephole_lstm_cell(self, x, c, h):
        i = tf.sigmoid(tf.matmul(x, self._Wi) + tf.matmul(h, self._Vi) + self._bi)
        f = tf.sigmoid(tf.matmul(x, self._Wf) + tf.matmul(h, self._Vf) + self._bf)
        o = tf.sigmoid(tf.matmul(x, self._Wo) + tf.matmul(h, self._Vo) + self._bo)
        z = tf.sigmoid(tf.matmul(x, self._Wz) + tf.matmul(h, self._Vz) + self._bz)
        c = f*c + i*z
        h = o*tf.tanh(c)
        o = tf.nn.xw_plus_b(o, self._W, self._b)
        return o, c, h
    
    # SCRN cell definition
    def _scrn_cell(self, x, c, h):
        h = tf.sigmoid(tf.matmul(c, self._P) + tf.matmul(x, self._A) + tf.matmul(h, self._R))
        c = (1 - self._alpha) * tf.matmul(x, self._B) + self._alpha * c
        o = tf.matmul(h, self._U) + tf.matmul(c, self._V) 
        return o, c, h
    
    # SRN cell definition
    def _srn_cell(self, x, c, h):
        h = tf.sigmoid(tf.matmul(x, self._A) + tf.matmul(h, self._R))
        o = tf.matmul(h, self._U)
        return o, c, h
    
    # RNN parameters:
    
    # Setup GRU cell parameters
    def _setup_gru_cell_parameters(self):
        self._Wz = tf.Variable(tf.truncated_normal([self._vocabulary_size, self._h_size], -0.1, 0.1))
        self._Vz = tf.Variable(tf.truncated_normal([self._h_size, self._h_size], -0.1, 0.1))
        self._bz = tf.Variable(tf.zeros([1, self._h_size]))
        self._Wr = tf.Variable(tf.truncated_normal([self._vocabulary_size, self._h_size], -0.1, 0.1))
        self._Vr = tf.Variable(tf.truncated_normal([self._h_size, self._h_size], -0.1, 0.1))
        self._br = tf.Variable(tf.zeros([1, self._h_size]))
        self._Wh = tf.Variable(tf.truncated_normal([self._vocabulary_size, self._h_size], -0.1, 0.1))
        self._Vh = tf.Variable(tf.truncated_normal([self._h_size, self._h_size], -0.1, 0.1))
        self._bh = tf.Variable(tf.zeros([1, self._h_size]))
        self._W = tf.Variable(tf.truncated_normal([self._h_size, self._vocabulary_size], -0.1, 0.1))
        self._b = tf.Variable(tf.zeros([self._vocabulary_size]))
    
    # Setup Peephole LSTM cell parameters
    def _setup_peephole_lstm_cell_parameters(self):
        self._Wi = tf.Variable(tf.truncated_normal([self._vocabulary_size, self._h_size], -0.1, 0.1))
        self._Vi = tf.Variable(tf.truncated_normal([self._h_size, self._h_size], -0.1, 0.1))
        self._bi = tf.Variable(tf.zeros([1, self._h_size]))
        self._Wf = tf.Variable(tf.truncated_normal([self._vocabulary_size, self._h_size], -0.1, 0.1))
        self._Vf = tf.Variable(tf.truncated_normal([self._h_size, self._h_size], -0.1, 0.1))
        self._bf = tf.Variable(tf.zeros([1, self._h_size]))
        self._Wo = tf.Variable(tf.truncated_normal([self._vocabulary_size, self._h_size], -0.1, 0.1))
        self._Vo = tf.Variable(tf.truncated_normal([self._h_size, self._h_size], -0.1, 0.1))
        self._bo = tf.Variable(tf.zeros([1, self._h_size]))
        self._Wz = tf.Variable(tf.truncated_normal([self._vocabulary_size, self._h_size], -0.1, 0.1))
        self._Vz = tf.Variable(tf.truncated_normal([self._h_size, self._h_size], -0.1, 0.1))
        self._bz = tf.Variable(tf.zeros([1, self._h_size]))
        self._W = tf.Variable(tf.truncated_normal([self._h_size, self._vocabulary_size], -0.1, 0.1))
        self._b = tf.Variable(tf.zeros([self._vocabulary_size]))
        
    # Setup SCRN cell parameters
    def _setup_scrn_cell_parameters(self):
        self._B = tf.Variable(tf.truncated_normal([self._vocabulary_size, self._c_size], -0.1, 0.1))
        self._A = tf.Variable(tf.truncated_normal([self._vocabulary_size, self._h_size], -0.1, 0.1))
        self._P = tf.Variable(tf.truncated_normal([self._c_size, self._h_size], -0.1, 0.1))
        self._R = tf.Variable(tf.truncated_normal([self._h_size, self._h_size], -0.1, 0.1))
        self._U = tf.Variable(tf.truncated_normal([self._h_size, self._vocabulary_size], -0.1, 0.1))
        self._V = tf.Variable(tf.truncated_normal([self._c_size, self._vocabulary_size], -0.1, 0.1))
        
    # Setup SRN cell parameters
    def _setup_srn_cell_parameters(self):
        self._A = tf.Variable(tf.truncated_normal([self._vocabulary_size, self._h_size], -0.1, 0.1))
        self._R = tf.Variable(tf.truncated_normal([self._h_size, self._h_size], -0.1, 0.1))
        self._U = tf.Variable(tf.truncated_normal([self._h_size, self._vocabulary_size], -0.1, 0.1))
        
    # Wrapper function to set cell being used
    def _cell_wrapper(self, x, c, h):
        if self._cell_flag == 1:
            o, c, h = self._gru_cell(x, c, h)
        elif self._cell_flag == 2:
            o, c, h = self._peephole_lstm_cell(x, c, h)
        elif self._cell_flag == 3:
            o, c, h = self._scrn_cell(x, c, h)
        elif self._cell_flag == 4:
            o, c, h = self._srn_cell(x, c, h)
        return o, c, h
    
    # Wrapper function to setup cell being used
    def _setup_cell_parameters_wrapper(self):
        if self._cell_flag == 1:
            self._setup_gru_cell_parameters()
        if self._cell_flag == 2:
            self._setup_peephole_lstm_cell_parameters()
        elif self._cell_flag == 3:
            self._setup_scrn_cell_parameters()
        elif self._cell_flag == 4:
            self._setup_srn_cell_parameters()
    
    # Implements a tower to run part of a batch of training data on a GPU
    def _training_tower(self, i, tower, gpu):
        
        with tf.device("/job:localhost/task:0/replica:0/gpu:%d" % gpu):
            with tf.name_scope('tower_%d' % tower) as scope:
        
                # Get saved training state
                c = self._training_c_saved[tower]
                h = self._training_h_saved[tower]

                # Run training data through LSTM cells
                labels = []
                outputs = []
                for j in range(self._optimization_frequency):
                    x = self._training_data[tower][i*self._optimization_frequency + j]
                    label = self._training_data[tower][i*self._optimization_frequency + j + 1]
                    o, c, h = self._cell_wrapper(x, c, h)
                    labels.append(label)
                    outputs.append(o)

                # Save training state and return training outputs
                with tf.control_dependencies([self._training_c_saved[tower].assign(c),
                                              self._training_h_saved[tower].assign(h)]):
                    return outputs, labels
        
    # Implements a tower to run part of a batch of validation data on a GPU
    def _validation_tower(self, tower, gpu):
        
        with tf.device("/job:localhost/task:0/replica:0/gpu:%d" % gpu):
            with tf.name_scope('tower_%d' % tower) as scope:
        
                # Get saved validation state
                c = self._validation_c_saved[tower]
                h = self._validation_h_saved[tower]

                # Run validation data through LSTM cells
                outputs = []
                for i in range(self._num_validation_unfoldings):
                    x = self._validation_input[tower][i]
                    o, c, h = self._cell_wrapper(x, c, h)
                    outputs.append(o)

                # Save validation state and return validation outputs
                with tf.control_dependencies([self._validation_c_saved[tower].assign(c), 
                                              self._validation_h_saved[tower].assign(h)]):
                    return outputs
            
    # Train model parameters
    def train(self, learning_rate, learning_decay, num_epochs, summary_frequency, training_text, validation_text):

        # Generate training batches
        print('Training Batch Generator:')
        training_batches = []
        for tower in range(self._num_towers):
            training_batches.append(batch_generator(tower, training_text[tower], self._batch_size,
                                                    self._num_training_unfoldings, self._vocabulary_size))
        
        # Generate validation batches
        print('Validation Batch Generator:')
        validation_batches = []
        tower = 0
        for tower in range(self._num_towers):
            validation_batches.append(batch_generator(tower, validation_text[tower], 1,
                                                      self._num_validation_unfoldings, self._vocabulary_size))
        
        # Training loop
        batch_ctr = 0
        epoch_ctr = 0
        training_feed_dict = dict()
        validation_feed_dict = dict()
        with tf.Session(graph=self._graph, config=tf.ConfigProto(log_device_placement=True)) as session:
        
            session.run(self._initialization)
            print('Initialized')

            # Iterate over fixed number of training epochs
            for epoch in range(num_epochs):

                # Display the learning rate for this epoch
                print('Epoch: %d  Learning Rate: %.2f' % (epoch+1, learning_rate))

                # Training Step:

                # Iterate over training batches
                for tower in range(self._num_towers):
                    training_batches[tower].reset_token_idx()
                session.run(self._reset_training_state)
                for batch in range(training_batches[0].num_batches()):

                    # Get next training batch
                    training_batches_next = []
                    tower = 0
                    for tower in range(self._num_towers):
                        training_batches_next.append([])
                        training_batches_next[tower] = training_batches[tower].next()
                    batch_ctr += 1

                    # Optimization
                    training_feed_dict[self._learning_rate] = learning_rate
                    for tower in range(self._num_towers):
                        for i in range(self._num_training_unfoldings + 1):
                            training_feed_dict[self._training_data[tower][i]] = training_batches_next[tower][i]
                    session.run(self._optimize, feed_dict=training_feed_dict)

                    # Summarize current performance
                    if (batch+1) % summary_frequency == 0:
                        cst = session.run(self._cost, feed_dict=training_feed_dict)
                        print('     Total Batches: %d  Current Batch: %d  Cost: %.2f' % 
                              (batch_ctr, batch+1, cst))
                      
                # Validation Step:
        
                # Iterate over validation batches
                for tower in range(self._num_towers):
                    validation_batches[tower].reset_token_idx()
                session.run(self._reset_validation_state)
                validation_log_prob_sum = 0
                for _ in range(validation_batches[0].num_batches()):
                    
                    # Get next validation batch
                    validation_batches_next = []
                    tower = 0
                    for tower in range(self._num_towers):
                        validation_batches_next.append([])
                        validation_batches_next[tower] = validation_batches[tower].next()
                    
                    # Validation
                    validation_batches_next_label = []
                    for tower in range(self._num_towers):
                        validation_batches_next_label_tmp = []
                        for i in range(self._num_validation_unfoldings):
                            validation_feed_dict[self._validation_input[tower][i]] = validation_batches_next[tower][i]
                            validation_batches_next_label_tmp.append(validation_batches_next[tower][i+1])
                        validation_batches_next_label.append(validation_batches_next_label_tmp)
                    validation_prediction = session.run(self._validation_prediction, feed_dict=validation_feed_dict)
                    
                    # Summarize current performance
                    for tower in range(self._num_towers):
                        for i in range(self._num_validation_unfoldings):
                            validation_log_prob_sum = validation_log_prob_sum + \
                                log_prob(validation_prediction[tower][i], validation_batches_next_label[tower][i])
                    
                # Calculation validation perplexity
                N = self._num_towers*self._num_validation_unfoldings*validation_batches[0].num_batches()
                perplexity = float(2 ** (-validation_log_prob_sum / N))
                print('Epoch: %d  Validation Set Perplexity: %.2f' % (epoch+1, perplexity))

                # Update learning rate
                if epoch > 0 and perplexity > perplexity_last_epoch:
                    learning_rate *= learning_decay
                perplexity_last_epoch = perplexity