# use tf.data

# -*- coding: utf-8 -*-
"""nlp_estimators.ipynb
Automatically generated by Colaboratory.
Original file is located at
    https://colab.research.google.com/drive/1oXjNYSJ3VsRvAsXN4ClmtsVEgPW_CX_c
Classifying text with TensorFlow Estimators
===
This notebook demonstrates how to tackle a text classification problem using custom TensorFlow estimators, embeddings and the [tf.layers](https://www.tensorflow.org/api_docs/python/tf/layers) module. Along the way we'll learn about word2vec and transfer learning as a technique to bootstrap model performance when labeled data is a scarce resource.
## Setup
Let's begin importing the libraries we'll need. This notebook runs in Python 3 and TensorFlow v1.4 or more, but it can run in earlier versions of TensorFlow by replacing some of the import statements to the corresponding paths inside the `contrib` module.
### The IMDB Dataset
The dataset we wil be using is the IMDB [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/), which consists of $25,000$ highly polar movie reviews for training, and $25,000$ for testing. We will use this dataset to train a binary classifiation model, able to predict whether a review is positive or negative.
"""

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

tf.logging.set_verbosity(tf.logging.INFO)

batchsize = 100
vocab_size = 5000
sentence_size = 200  # 500
embedding_size = 50  # 32
nHidden = 100
EPOCHS = 200
nClasses = 2
model_dir = tempfile.mkdtemp()

"""### Loading the data
Keras provides a convenient handler for importing the dataset which is also available as a serialized numpy array `.npz` file to download [here]( https://s3.amazonaws.com/text-datasets/imdb.npz). Each review consists of a series of word indexes that go from $4$ (the most frequent word in the dataset, **the**) to $4999$, which corresponds to **orange**. Index $1$ represents the beginning of the sentence and the index $2$ is assigned to all unknown (also known as *out-of-vocabulary* or *OOV*) tokens. These indexes have been obtained by pre-processing the text data in a pipeline that cleans, normalizes and tokenizes each sentence first and then builds a dictionary indexing each of the tokens by frequency. We are not convering these techniques in this post, but you can take a look at [this chapter](http://www.nltk.org/book/ch03.html) of the NLTK book to learn more.
"""

(x_train_variable, y_train), (x_test_variable, y_test) = imdb.load_data(
    num_words=vocab_size)
# print(len(y_train), "train sequences")
# print(len(y_test), "test sequences")
y_train = keras.utils.to_categorical(y_train)

# pad data
# https://stackoverflow.com/questions/46298793/how-does-choosing-between-pre-and-post-zero-padding-of-sequences-impact-results
x_train = sequence.pad_sequences(x_train_variable, maxlen=sentence_size)
x_test = sequence.pad_sequences(x_test_variable, maxlen=sentence_size)
# print("x_train shape:", x_train.shape)
# print("x_test shape:", x_test.shape)

""" Embeddings
The next step of complexity we can add are word embeddings. Embeddings are a dense low-dimensional representation of sparse high-dimensional data. This allows our model to learn a more meaningful representation of each token, rather than just an index. While an individual dimension is not meaningful, the low-dimensional space---when learned from a large enough corpus---has been shown to capture relations such as tense, plural, gender, thematic relatedness, and many more. We can add word embeddings by converting our existing feature column into an `embedding_column`. The representation seen by the model is the mean of the embeddings for each token (see the `combiner` argument in the [docs](https://www.tensorflow.org/api_docs/python/tf/feature_column/embedding_column)). We can plug in the embedded features into a pre-canned `DNNClassifier`. 
A note for the keen observer: an `embedding_column` is just an efficient way of applying a fully connected layer to the sparse binary feature vector of tokens, which is multiplied by a constant depending on the chosen combiner. A direct consequence of this is that it wouldn't make sense to use an `embedding_column` directly in a `LinearClassifier` because two consecutive linear layers without non-linearities in between add no prediction power to the model, unless of course the embeddings are pre-trained.
"""
column = tf.feature_column.categorical_column_with_identity('x', vocab_size)
word_embedding_column = tf.feature_column.embedding_column(column, dimension=embedding_size)


""" Build tensor from array
There's one more thing we need to do get our data ready for TensorFlow. We need to convert the data from numpy arrays into Tensors. Fortunately for us the `Dataset` module has us covered. 
It provides a handy function, `from_tensor_slices` that creates the dataset to which we can then apply multiple transformations to shuffle, batch and repeat samples and plug into our training pipeline. Moreover, with just a few changes we could be loading the data from files on disk and the framework does all the memory management.
We define two input functions: one for processing the training data and one for processing the test data. We shuffle the training data and do not predefine the number of epochs we want to train, while we only need one epoch of the test data for evaluation. We also add an additional `"len"` key to both that captures the length of the original, unpadded sequence, which we will use later.
"""
"""
The tf.data API enables you to build complex input pipelines
https://www.tensorflow.org/guide/data
"""
# length of all sentences
x_len_train = np.array([min(len(x), sentence_size) for x in x_train_variable])
x_len_test = np.array([min(len(x), sentence_size) for x in x_test_variable])


def parser(x, length, y):
    features = {"x": x, "len": length}
    return features, y

dataset = tf.data.Dataset.from_tensor_slices((x_train, x_len_train, y_train))
dataset = dataset.shuffle(buffer_size=len(x_train_variable))
dataset = dataset.batch(batchsize)
# The Dataset.map(f) transformation produces a new dataset by
# applying a given function f to each element of the input dataset
dataset = dataset.map(parser)
# training workflow:  .repeat: process multiple epochs of the same data
dataset = dataset.repeat()
# https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428 for iterator and get_next
iterator = dataset.make_one_shot_iterator()
# call get_next() to get the tensor that will contain your data
# """
# el = iterator.get_next()
# with tf.Session() as sess:
#     print(sess.run(el))
# """
# return iterator.get_next()
feature_input, y_input = iterator.get_next()



def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((x_test, x_len_test, y_test))
    dataset = dataset.batch(batchsize)
    dataset = dataset.map(parser)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


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


def RNN(feature_input, weights, biases):
    inputs = tf.contrib.layers.embed_sequence(
        feature_input['x'], vocab_size, embedding_size,
        initializer=tf.random_uniform_initializer(-1.0, 1.0))

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(nHidden)

    outputs, final_states = tf.nn.dynamic_rnn(
        lstm_cell, inputs, sequence_length=feature_input['len'], dtype=tf.float32)

    return tf.multiply(weights['out'], tf.expand_dims(outputs[-1][-1], 1)) + biases['out']

pred = tf.nn.softmax(RNN(feature_input, weights, biases))

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=pred))

optimizer = tf.train.AdamOptimizer(1e-5).minimize(cost)

# ----------------
# other useful tensors
# -----------------
correctPred = tf.equal(tf.argmax(pred, 1), tf.argmax(y_input, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))


'''
Run
'''


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(EPOCHS):
        _, loss_value = sess.run([optimizer, cost])
        print("Iter: {}, Loss: {:.4f}".format(i, loss_value))

