# use keras

import numpy
import keras
import math
import tensorflow as tf
from matplotlib import pyplot
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.preprocessing import sequence

sess = keras.backend.get_session()

numpy.random.seed(7)
batchsize = 100
epoch_num = 3
embedding_vector_length = 32
sentence_size = 200
vocab_size = 5000
nHidden = 100


### prepare data
# load the dataset but only keep the top n words, zero the rest
top_words = vocab_size
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
one_hot_labels_train = keras.utils.to_categorical(y_train, num_classes=2)
# truncate and pad input sequences
max_review_length = sentence_size
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


### create the model
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(LSTM(nHidden))
# model.add(Dense(1, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

### Add AVH

### save AVH

class AVHHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.avh = []

    def on_batch_end(self, batch, logs={}):
        # weights = model.layers[2].weights
        all_weights = [layer.get_weights() for layer in model.layers]
        last_layer_weights = all_weights[2]
        weights = last_layer_weights[0]

        # avh = math.acos(weights)
        # avh = tf.acos(weights)
        # modelWeights = []
        # for layer in model.layers:
        #     layerWeights = []
        #     for weight in layer.get_weights():
        #         layerWeights.append(tf.acos(weight))
        #     modelWeights.append(layerWeights)
        # self.avh.append(modelWeights)
        self.avh.append(weights)

avhHistory = AVHHistory()

### save loss
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.trainacc = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.trainacc.append(logs.get('accuracy'))

losshistory = LossHistory()


### fit model
print(model.summary())
# history = model.fit(X_train, y_train, epochs=3, batch_size=64)
history = model.fit(X_train, one_hot_labels_train, epochs=epoch_num, batch_size=batchsize,
                    callbacks=[losshistory, avhHistory])

# Final evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=0)
one_hot_labels_test = keras.utils.to_categorical(y_test, num_classes=2)
scores = model.evaluate(X_test, one_hot_labels_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

pyplot.plot(history.history['accuracy'])
pyplot.show()

pyplot.plot(losshistory.losses)
pyplot.show()
pyplot.plot(losshistory.trainacc)
pyplot.show()

a=avhHistory.avh
b=a[1][1][1]