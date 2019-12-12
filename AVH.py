import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# -----------
# parameters
# --------
learningRate = 0.001
epochs = 16
batchSize = 100
displayStep = 1000

nInput = 28  # we want the input to take the 28 pixels
nSteps = 28  # every 28
nHidden = 64  # number of neurons for the RNN
nClasses = 10  # this is MNIST so you know


# --------------
# functions
# --------------
def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y


# -------------
# place holders
# -------------
x = tf.placeholder('float', [None, nSteps, nInput])
y = tf.placeholder('float', [None, nClasses])

# ----------------
# variables & model
# --------------------

# weights and biases from hidden layer to output:
weights = {
    'out': tf.get_variable('W', dtype=tf.float32, shape=[nHidden, nClasses],
                           initializer=tf.truncated_normal_initializer(stddev=0.01))
}

biases = {
	'out': tf.Variable(tf.random_normal([nClasses]))
}


def RNN(x, weights, biases):
    """ input: one image; output: 10 probabilities """

    # x = tf.transpose(x, [1,0,2])
    # x = tf.reshape(x, [-1, nInput])
    # x = tf.split(0, nSteps, x) #configuring so you can get it as needed for the 28 pixels

    # Unpacks num tensors from value by chipping it along the axis dimension. now x is 28 tensors
    x = tf.unstack(x, num=nSteps, axis=1)

    # lstmCell = rnn_cell.LSTMCell(nHidden)  # use LSTM
    # basicCell = rnn_cell.BasicRNNCell(nHidden)
    basicCell = rnn_cell.GRUCell(nHidden)

    # outputs, states = rnn.static_rnn(lstmCell, x, dtype=tf.float32)
    outputs, states = rnn.static_rnn(basicCell, x, dtype=tf.float32)
    # https://stats.stackexchange.com/questions/330176/what-is-the-output-of-a-tf-nn-dynamic-rnn

    return tf.matmul(outputs[-1], weights['out']) + biases['out']


pred = tf.nn.softmax(RNN(x, weights, biases))

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))

# ---------------
# optimizer
# ---------------------
optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)

# ----------------
# other useful tensors
# -----------------
correctPred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

# ----------------
# Data
# ------------------
x_train, y_train, x_valid, y_valid = mnist.train.images, mnist.train.labels, \
                                     mnist.validation.images, mnist.validation.labels
testData = mnist.test.images.reshape((-1, nSteps, nInput))
testLabel = mnist.test.labels


# -------------------
# run
# ----------------
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 1

    train_accuracy = []
    test_accuracy = []
    train_loss = []

    x_train, y_train = randomize(x_train, y_train)
    num_tr_iter = int(len(y_train) / batchSize)

    global_step = 0

    for epoch in range(epochs):

        for iteration in range(num_tr_iter):
            global_step += 1
            start = iteration * batchSize
            end = (iteration + 1) * batchSize
            batchX = x_train[start:end]
            batchY = y_train[start:end]
            batchX = batchX.reshape((batchSize, nSteps, nInput))

            sess.run(optimizer, feed_dict={x: batchX, y: batchY})

            if global_step % displayStep == 0:
                train_acc = sess.run(accuracy, feed_dict={x: batchX, y: batchY})
                train_accuracy.append(train_acc)
                test_acc = sess.run(accuracy, feed_dict={x: testData, y: testLabel})
                test_accuracy.append(test_acc)
                loss = sess.run(cost, feed_dict={x: batchX, y: batchY})
                train_loss.append(loss)
                print("Iter " + str(global_step) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(train_acc) + ", Test Accuracy= " + \
                      "{:.5f}".format(test_acc))  # {:.6f}.format specify decimal places


    with open('Q2_accuracy_and_loss_LSTM.txt', 'w') as f:
        f.write(",".join([str(x) for x in train_accuracy]) + "\n")
        f.write(",".join([str(x) for x in test_accuracy]) + "\n")
        f.write(",".join([str(x) for x in train_loss]))

    print('Optimization finished')

    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: testData, y: testLabel}))
