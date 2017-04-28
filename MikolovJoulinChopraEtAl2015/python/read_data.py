# usecase_flg = 1 for predicting letters
#               2 for predicting words with fixed vocabulary size
#               3 for predicting words with cutoff for infrequent words

# Imports
import tensorflow as tf
import zipfile

def read_data(usecase_flg, filename, vocabulary_size):
    # read datafile
    with zipfile.ZipFile(filename) as f:
        if usecase_flg == 1:
            raw_data = tf.compat.as_str(f.read(f.namelist()[0]))
        elif usecase_flg >= 2:
            raw_data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    # Return data
    return raw_data