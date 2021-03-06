from load import load_data
import tensorflow as tf
import numpy as np
import random
import sklearn
from sklearn import preprocessing

# fixing seed for reproduction of results
np_rand = random.randint(0,10000)
from numpy.random import seed
seed(np_rand)

tf_rand = random.randint(0,10000)
from tensorflow import set_random_seed
set_random_seed(tf_rand)

print('np seed: ', np_rand)
print('tf seed: ', tf_rand)

# number of variables we consider in each input
num_vars = 14

# after some searching, these seem to be optimal values
iter_ = 20000
lr = 1e-1
batch_size = 32
fc1_size = 200
fc2_size = 200

init = tf.initialize_all_variables()

# input placeholder
X = tf.placeholder(tf.float32, [None, num_vars])

# y_true
Y_ = tf.placeholder(tf.float32, [None, 1])

# fc1
W_1 = tf.Variable(tf.truncated_normal([num_vars, fc1_size]))
b_1 = tf.Variable(tf.truncated_normal([fc1_size]))
fc1 = tf.nn.softmax(tf.matmul(X, W_1) + b_1)

# fc2 
W_2 = tf.Variable(tf.truncated_normal([fc1_size,fc2_size]))
b_2 = tf.Variable(tf.truncated_normal([fc2_size]))
fc2 = tf.nn.softmax(tf.matmul(fc1, W_2) + b_2)

# fc3
W_3 = tf.Variable(tf.truncated_normal([fc2_size,1]))
b_3 = tf.Variable(tf.truncated_normal([1]))
out = tf.matmul(fc2, W_3) + b_3

cost = tf.reduce_mean(tf.square(out - Y_))
train = tf.train.AdamOptimizer(lr).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    
    print('Data snippit...')
    # load same data as cvx fit
    x_train, y_train = load_data('air_quality_train')
    x_test, y_test = load_data('air_quality_test')
    
    print(x_train[0])
    print(y_train[0])

    # convert y_train to tf usable shape (not (148,) np shape)
    y_train = np.reshape(y_train, [len(y_train),1])

    # normalize input data
    x_train = sklearn.preprocessing.scale(x_train)
    x_test = sklearn.preprocessing.scale(x_test)

    for i in range(iter_):
        # sklearn shuffle to get minibatch
        shuffled_x, shuffled_y = sklearn.utils.shuffle(x_train, y_train, n_samples=batch_size)

        # train
        _, current_cost = sess.run([train, cost], feed_dict={X: shuffled_x, Y_: shuffled_y})

        # apply weight clipping
        #sess.run(clip)

        # printing progress
        if i % 100 == 0:
            print('Iter: {0} Train mse: {1}'.format(i, current_cost))

# ======================================================================= #
    # print MSE on entire training set
    print('\nTesting on train-set...')
    pred = sess.run(out, feed_dict={X: x_train})
    error = []

    #print('{0:25} {1}'.format('pred', 'real'))
    for j in range(len(pred)):
        #print('{0:<25} {1}'.format(str(pred[j][0]), y_test[j]))
        error.append((y_train[j] - pred[j][0]) ** 2)

    print('\nTotal train size: ', len(y_train))
    print('Train mse: ', sum(error) / len(error))

# ======================================================================== #
    # test on validation set
    print('\nTesting on validation-set...')
    pred = sess.run(out, feed_dict={X: x_test})
    error = []

    #print('{0:25} {1}'.format('pred', 'real'))
    for j in range(len(pred)):
        #print('{0:<25} {1}'.format(str(pred[j][0]), y_test[j]))
        error.append((y_test[j] - pred[j][0]) ** 2)

    print('\nTotal test size: ', len(y_train))
    print('Test mse: ', sum(error) / len(error))
