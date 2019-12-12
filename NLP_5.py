# track AVH

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

np.random.seed(7)
stepsize=1e-3
batchSize = 100
vocab_size = 5000
sentence_size = 200
embedding_size = 50
nHidden = 100
epochs = 6
nClasses = 2
displayStep = 500


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
The next step of complexity we can add are word embeddings. Embeddings are a dense low-dimensional representation of sparse high-dimensional data. This allows our model to learn a more meaningful representation of each token, rather than just an index. While an individual dimension is not meaningful, the low-dimensional space---when learned from a large enough corpus---has been shown to capture relations such as tense, plural, gender, thematic relatedness, and many more. We can add word embeddings by converting our existing feature column into an `embedding_column`. The representation seen by the model is the mean of the embeddings for each token (see the `combiner` argument in the [docs](https://www.tensorflow.org/api_docs/python/tf/feature_column/embedding_column)). We can plug in the embedded features into a pre-canned `DNNClassifier`. 
A note for the keen observer: an `embedding_column` is just an efficient way of applying a fully connected layer to the sparse binary feature vector of tokens, which is multiplied by a constant depending on the chosen combiner. A direct consequence of this is that it wouldn't make sense to use an `embedding_column` directly in a `LinearClassifier` because two consecutive linear layers without non-linearities in between add no prediction power to the model, unless of course the embeddings are pre-trained.
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
weights = {
    'out': tf.get_variable('W', dtype=tf.float32, shape=[nHidden, nClasses],
                           initializer=tf.truncated_normal_initializer(stddev=0.01))
}
biases = {
    'out': tf.Variable(tf.random_normal([nClasses]))
}


inputs = tf.contrib.layers.embed_sequence(
    x, vocab_size, embedding_size,
    initializer=tf.random_uniform_initializer(-1.0, 1.0))

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(nHidden)

outputs, final_states = tf.nn.dynamic_rnn(
    lstm_cell, inputs, sequence_length=x_len, dtype=tf.float32)

logits = tf.matmul(final_states.h, weights['out']) + biases['out']

# cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

# optimizer
optimizer = tf.train.AdamOptimizer().minimize(cost)


"""
# other useful tensors
"""
correctPred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
# -------------
# AVH
# --------------
wy = tf.tensordot(y, weights['out'], axes=((1,), (1,)))
# wy = tf.matmul(weights['out'], tf.transpose(y))
# wy = tf.transpose(wy)
dotprod = tf.reduce_sum(tf.multiply(final_states.h, wy), axis=1)
xnorm = tf.norm(final_states.h, axis=1)
wynorm = tf.norm(wy, axis=1)
num = tf.acos(tf.divide(tf.divide(dotprod, xnorm), wynorm))
dotprod = tf.tensordot(final_states.h, weights['out'], axes=((1,), (0,)))
wnorm = tf.broadcast_to(tf.norm(weights['out'], axis=0), shape=[batchSize, nClasses])
xnorm = tf.reshape(xnorm, shape=[-1, 1])
xnormb = tf.broadcast_to(xnorm, shape=[batchSize, nClasses])
denom = tf.acos(tf.divide(tf.divide(dotprod, wnorm), xnormb))
denom = tf.reduce_sum(denom, axis=1)
avh = tf.divide(num, denom)
avh_mean = tf.reduce_mean(avh)


'''
Run
'''
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    train_accuracy = []
    test_accuracy = []
    train_loss = []
    train_AVH_mean = []

    # x_train, y_train, x_len_train = randomize(x_train, y_train, x_len_train)
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
                train_acc = sess.run(accuracy, feed_dict={x: batchX, y: batchY, x_len: batchX_len})
                train_accuracy.append(train_acc)
                test_acc = sess.run(accuracy, feed_dict={x: x_test, y: y_test, x_len: x_len_test})
                test_accuracy.append(test_acc)
                loss = sess.run(cost, feed_dict={x: batchX, y: batchY, x_len: batchX_len})
                train_loss.append(loss)
                train_avh = sess.run(avh_mean, feed_dict={x: batchX, y: batchY, x_len: batchX_len})
                train_AVH_mean.append(train_avh)
                print("Iter " + str(global_step) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(train_acc) + ", Test Accuracy= " + \
                      "{:.5f}".format(test_acc) + ", Training AVH Mean= " + \
                      "{:.5f}".format(train_avh))  # {:.6f}.format specify decimal places

    with open('NLP_5.txt', 'w') as f:
        f.write(",".join([str(x) for x in train_AVH_mean]) + "\n")
        f.write(",".join([str(x) for x in train_accuracy]) + "\n")
        f.write(",".join([str(x) for x in test_accuracy]) + "\n")
        f.write(",".join([str(x) for x in train_loss]))

    print('Optimization finished')

    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: x_test, y: y_test, x_len: x_len_test}))
