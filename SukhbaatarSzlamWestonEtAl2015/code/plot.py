import matplotlib.pyplot as plt
import pickle
import sys
vocab_location = "/home/ashwin/WorkBench/LatentlyDeepLearningCertificate/SukhbaatarSzlamWestonEtAl2015/history/memory/"
filename = "0" if len(sys.argv)==1 else sys.argv[1]
fullname = vocab_location + filename + ".p"
history = pickle.load( open( fullname, "rb" ) )
print(history.keys())
# summarize history for accuracy
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for precision
plt.plot(history['precision'])
plt.plot(history['val_precision'])
plt.title('model precision')
plt.ylabel('precision')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for recall
plt.plot(history['recall'])
plt.plot(history['val_recall'])
plt.title('model recall')
plt.ylabel('recall')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for fmeasure
plt.plot(history['fmeasure'])
plt.plot(history['val_fmeasure'])
plt.title('model fmeasure')
plt.ylabel('fmeasure')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
