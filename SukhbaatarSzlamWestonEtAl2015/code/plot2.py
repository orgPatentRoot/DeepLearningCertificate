import matplotlib.pyplot as plt
import pickle
import sys
from keras.models import load_model

filename = "0" if len(sys.argv)==1 else sys.argv[1]
check_point_location = "/home/ashwin/WorkBench/LatentlyDeepLearningCertificate/SukhbaatarSzlamWestonEtAl2015/model/attention/" + filename + ".hdf5"
model = load_model(check_point_location)
history = model.history.history
print(history.keys())
