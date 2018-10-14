from load import load_data
import tensorflow as tf
import numpy as np
import random
import sklearn
from sklearn import preprocessing
# ----------------------------------------------
#Adding seed for reproduction of results
np_rand = random.randint(0,10000)
from numpy.random import seed
seed(np_rand)

tf_rand = random.randint(0,10000)
from tensorflow import set_random_seed
set_random_seed(tf_rand)

print('np seed: ', np_rand)
print('tf seed: ', tf_rand)
# ----------------------------------------------

def proj_weights(W):
    unpacked = tf.unstack(W)
    new_tensor = []

    for w in unpacked:
        result_tensor = tf.math.maximum(w, 0)
        new_tensor.append(result_tensor)

    output = tf.stack(new_tensor)

    return output

# number of variables we consider in each input
num_vars = 5

print('Data snippit...')
# load same data as cvx fit
x_train, y_train, x_test, y_test = load_data()

# so that we have a solid baseline, only use training data that cvx fit uses
num_train = 100
x_train = x_train[:num_train]
y_train = y_train[:num_train]

# normalize input data
x_train = sklearn.preprocessing.scale(x_train)
x_test = sklearn.preprocessing.scale(x_test)

# convert y_train to tf usable shape (not (148,) np shape)
y_train = np.reshape(y_train, [len(y_train),1])

# after some searching, these seem to be optimal values
iter_ = 1000
lr = 1e-2
batch_size = 32
fc1_size = 32

init = tf.initialize_all_variables()

# input placeholder
X = tf.placeholder(tf.float32, [None, num_vars])

# y_true
Y_ = tf.placeholder(tf.float32, [None, 1])

# fc1
W_1 = tf.Variable(tf.truncated_normal([num_vars, fc1_size]))
b_1 = tf.Variable(tf.truncated_normal([fc1_size]))
fc1 = tf.nn.softmax(tf.matmul(X, W_1) + b_1)

# fc2 (out)
W_2 = tf.Variable(tf.truncated_normal([fc1_size,1]))
b_2 = tf.Variable(tf.truncated_normal([1]))
out = tf.matmul(fc1, W_2) + b_2

cost = tf.reduce_mean(tf.square(out - Y_))
train = tf.train.GradientDescentOptimizer(lr).minimize(cost)

# clipping op so weights stay in [0,inf)
clip_W_1 = W_1.assign(tf.maximum(0., W_1)) 
clip_W_2 = W_2.assign(tf.maximum(0., W_2)) 
clip = tf.group(clip_W_1, clip_W_2)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for i in range(iter_):
        
        # sklearn shuffle to get minibatch
        shuffled_x, shuffled_y = sklearn.utils.shuffle(x_train, y_train, n_samples=batch_size)

        # train
        _, current_cost = sess.run([train, cost], feed_dict={X: shuffled_x,
                                                             Y_: shuffled_y}) 
        sess.run(clip) 
        if i % 100 == 0:
            print('Iter: {0} Train mse: {1}'.format(i, current_cost))

    print('\nTesting on test-set...')
    pred = sess.run(out, feed_dict={X: x_test})
    error = []

    print('{0:25} {1}'.format('pred', 'real'))
    for j in range(len(pred)):
        print('{0:<25} {1}'.format(str(pred[j][0]), y_test[j]))
        error.append((y_test[j] - pred[j][0]) ** 2)

    print('\nTest batch size: ', len(y_test))
    print('Test mse: ', sum(error) / len(error))

    tvars = tf.trainable_variables()
    tvars_vals = sess.run(tvars)
    for var, val in zip(tvars, tvars_vals):
        print(var.name, val)  # Prints the name of the variable alongside its value.
