# Imports
import numpy as np

# Calculate the log-probability of the labels given predictions
def log_prob(predictions, labels):
    predictions[predictions < 1e-10] = 1e-10
    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]