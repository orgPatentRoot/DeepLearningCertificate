# Delta Recurrent Neural Network (Delta-RNN) Framework
#
# This gives an implementation of the Delta-RNN framework given in Ororbia et al. 2017, arXiv:1703.08864 [cs.CL], 
# https://arxiv.org/abs/1703.08864 using Python and Tensorflow.
#
# A read data function that reads data in a zip-file for feeding into the Delta-RNN models.
#
# Stuart Hagler, 2017

# usecase_flg = 1 for predicting letters
#               2 for predicting words with cutoff for infrequent words

# Imports
import tensorflow as tf
import zipfile

def read_data(usecase_flg, filename):
    # read datafile
    with zipfile.ZipFile(filename) as f:
        if usecase_flg == 1:
            raw_data = tf.compat.as_str(f.read(f.namelist()[0]))
        elif usecase_flg == 2:
            raw_data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    # Return data
    return raw_data