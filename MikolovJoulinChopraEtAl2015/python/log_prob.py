# Structurally Contrained Recurrent Network (SCRN) Model
#
# This gives an implementation of the SCRN model given in Mikolov et al. 2015, arXiv:1412.7753 [cs.NE], 
# https://arxiv.org/abs/1412.7753 using Python and Tensorflow.
#
# The log probability function used to calculate the validation perplexity the LSTM, SCRN, and SNN models.
#
# Stuart Hagler, 2017

# Imports
import numpy as np

# Calculate the log-probability of the labels given predictions
def log_prob(predictions, labels):
    predictions[predictions < 1e-10] = 1e-10
    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]