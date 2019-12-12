# AVH weighted loss for RNN

import os
import string
import tempfile
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import imdb
from tensorflow.python.keras.preprocessing import sequence
from tensorboard import summary as summary_lib

import keras

np.random.seed(3)
stepsize=1e-3
batchSize = 100
vocab_size = 5000
sentence_size = 200
embedding_size = 32
nHidden = 100
epochs = 2
nClasses = 2
displayStep = 10


"""
Loading the data
"""
(x_train_variable, y_train), (x_test_variable, y_test) = imdb.load_data(
    num_words=vocab_size)
# print(len(y_train), "train sequences")
# print(len(y_test), "test sequences")

# length of all sentences
x_len_train = np.array([min(len(x), sentence_size) for x in x_train_variable])
x_len_test = np.array([min(len(x), sentence_size) for x in x_test_variable])

# one-hot encoding
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# pad data
# https://stackoverflow.com/questions/46298793/how-does-choosing-between-pre-and-post-zero-padding-of-sequences-impact-results
x_train = sequence.pad_sequences(x_train_variable, maxlen=sentence_size)
x_test = sequence.pad_sequences(x_test_variable, maxlen=sentence_size)
# print("x_train shape:", x_train.shape)
# print("x_test shape:", x_test.shape)


"""
batch
"""
def randomize(x, y, x_len):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    shuffled_x_len = x_len[permutation]
    return shuffled_x, shuffled_y, shuffled_x_len


""" 
Embeddings
"""
column = tf.feature_column.categorical_column_with_identity('x', vocab_size)
word_embedding_column = tf.feature_column.embedding_column(column, dimension=embedding_size)

"""
Place holders to pass data
"""
x = tf.placeholder('int32', [None, sentence_size])
y = tf.placeholder('float', [None, nClasses])
x_len = tf.placeholder('int32', [None, ])

"""
LSTM Networks
"""
# weights and biases from hidden layer to output:
# weights = {
#     'out': tf.get_variable('W', dtype=tf.float32, shape=[nHidden, nClasses],
#                            initializer=tf.truncated_normal_initializer(stddev=0.01))
# }
weights = {
    'out': tf.get_variable('W', dtype=tf.float32, shape=[nHidden, nClasses],
                           initializer=tf.contrib.layers.xavier_initializer())
}

# biases = {
#     'out': tf.Variable(tf.random_normal([nClasses]))
# }
biases = tf.get_variable(name="b", shape=[nClasses], initializer=tf.zeros_initializer())

# hidden layers
# inputs = tf.contrib.layers.embed_sequence(
#     x, vocab_size, embedding_size,
#     initializer=tf.random_uniform_initializer(-1.0, 1.0))
inputs = tf.contrib.layers.embed_sequence(
    x, vocab_size, embedding_size,
    initializer=tf.contrib.layers.xavier_initializer())


# lstm_cell = tf.nn.rnn_cell.basicLSTMCell(nHidden)
lstm_cell = tf.nn.rnn_cell.LSTMCell(nHidden)
# lstm_cell = tf.keras.layers.LSTMCell(nHidden)

outputs, final_states = tf.nn.dynamic_rnn(
    lstm_cell, inputs, sequence_length=x_len, dtype=tf.float32)
# outputs, final_states = tf.keras.layers.RNN(
#     lstm_cell, x, dtype=tf.float32)

# logits = tf.matmul(final_states.h, weights['out']) + biases['out']
logits = tf.matmul(final_states.h, weights['out']) + biases

"""
# other useful tensors
"""
correctPred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
w_norm = tf.math.reduce_mean(
    tf.math.sqrt( tf.math.reduce_mean(weights['out']*weights['out'], axis=0)))
activation_norm = tf.reduce_mean(
    tf.math.sqrt( tf.math.reduce_mean(final_states.h*final_states.h, axis=1)))

# -------------
# AVH
# --------------
wy = tf.tensordot(y, weights['out'], axes=((1,), (1,)))
dotprod = tf.reduce_sum(tf.multiply(final_states.h, wy), axis=1)
xnorm = tf.norm(final_states.h, axis=1)
wynorm = tf.norm(wy, axis=1)
num = tf.acos(tf.divide(tf.divide(dotprod, xnorm), wynorm))
dotprod = tf.tensordot(final_states.h, weights['out'], axes=((1,), (0,)))
wnorm = tf.broadcast_to(tf.norm(weights['out'], axis=0), shape=[tf.shape(final_states.h)[0], nClasses])
xnorm = tf.reshape(xnorm, shape=[-1, 1])
xnormb = tf.broadcast_to(xnorm, shape=[tf.shape(final_states.h)[0], nClasses])
denom = tf.acos(tf.divide(tf.divide(dotprod, wnorm), xnormb))
denom = tf.reduce_sum(denom, axis=1)
avh = tf.divide(num, denom)
avh_mean = tf.reduce_mean(avh)


"""
cost and optimizer
"""
# cost
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
y_conv = tf.nn.softmax(logits, name='yscore')
cross_entropy = -tf.reduce_sum(y * tf.log(y_conv), reduction_indices=[1])
# cross_entropy = cross_entropy + tf.reduce_sum(avh*avh)
cross_entropy = cross_entropy*tf.exp(avh)
cost = tf.reduce_mean(cross_entropy)
# cost = cost + tf.reduce_mean(avh)

# optimizer
optimizer = tf.train.AdamOptimizer().minimize(cost)


'''
Run
'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_accuracy = []
    train_loss = []
    train_AVH_mean = []
    train_w_norm = []
    train_activation_norm = []
    test_accuracy = []
    test_loss = []
    test_AVH_mean = []
    test_activation_norm = []

    x_train, y_train, x_len_train = randomize(x_train, y_train, x_len_train)
    num_tr_iter = int(len(y_train) / batchSize)

    global_step = 0

    for epoch in range(epochs):

        for iteration in range(num_tr_iter):
            global_step += 1
            start = iteration * batchSize
            end = (iteration + 1) * batchSize
            batchX = x_train[start:end]
            batchY = y_train[start:end]
            batchX_len = x_len_train[start:end]

            sess.run(optimizer, feed_dict={x: batchX, y: batchY, x_len: batchX_len})

            if global_step % displayStep == 0:
                train_avh, train_l, train_acc, train_w_n, train_act_n = sess.run(
                    [avh_mean, cost, accuracy, w_norm, activation_norm],
                    feed_dict={x: batchX, y: batchY, x_len: batchX_len})
                train_loss.append(train_l)
                train_accuracy.append(train_acc)
                train_AVH_mean.append(train_avh)
                train_w_norm.append(train_w_n)
                train_activation_norm.append(train_act_n)
                test_avh, test_l, test_acc, test_act_n = sess.run(
                    [avh_mean, cost, accuracy, activation_norm],
                    feed_dict={x: x_test, y: y_test, x_len: x_len_test})
                test_loss.append(test_l)
                test_accuracy.append(test_acc)
                test_AVH_mean.append(test_avh)
                test_activation_norm.append(test_act_n)

                print("Iter " + str(global_step) + ", Minibatch Loss= " + \
                      "{:.6f}".format(train_l) + ", Test Accuracy= " + \
                      "{:.5f}".format(test_acc) + ", Test AVH Mean= " + \
                      "{:.5f}".format(test_avh))  # {:.6f}.format specify decimal places

    with open('NLP_mul_seed3.txt', 'w') as f:
        f.write(",".join([str(x) for x in train_loss]) + "\n")
        f.write(",".join([str(x) for x in train_accuracy]) + "\n")
        f.write(",".join([str(x) for x in train_AVH_mean]) + "\n")
        f.write(",".join([str(x) for x in train_w_norm]) + "\n")
        f.write(",".join([str(x) for x in train_activation_norm]) + "\n")
        f.write(",".join([str(x) for x in test_loss]) + "\n")
        f.write(",".join([str(x) for x in test_accuracy]) + "\n")
        f.write(",".join([str(x) for x in test_AVH_mean]) + "\n")
        f.write(",".join([str(x) for x in test_activation_norm]))

    print('Optimization finished')

    # print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: x_test, y: y_test, x_len: x_len_test}))

