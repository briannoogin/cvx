from load import load_data
import tensorflow as tf
import numpy as np

x_train, y_train, x_test, y_test = load_data()

print(x_train.shape)
#Adding Seed so that random initialization is consistent
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

num_vars = 8
iter_ = 100
lr = 1e-3

fc1_size = 10

init = tf.global_variables_initializer()

# input placeholder
X = tf.placeholder(tf.float32, [None, num_vars])

# true
Y_ = tf.placeholder(tf.float32, [None, 1])

# fc1
W_1 = tf.Variable(tf.truncated_normal([num_vars, fc1_size]))
b_1 = tf.Variable(tf.truncated_normal([fc1_size]))
fc1 = tf.nn.softmax(tf.matmul(X, W_1) + b_1)
fc1 = tf.nn.relu(fc1)

# fc2 (out)
W_2 = tf.Variable(tf.truncated_normal([fc1_size,1]))
b_2 = tf.Variable(tf.truncated_normal([1]))
out = tf.matmul(fc1, W_2) + b_2

cost = tf.reduce_mean(tf.square(out - Y_))
train = tf.train.GradientDescentOptimizer(lr).minimize(cost)

with tf.Session() as sess:
	sess.run(init)

	for i in range(iter_):
		sess.run(cost, feed_dict={X: x_train,
								  Y_: y_train})

	pred = sess.run(out, feed_dict={X: x_test})

	for j in range(len(pred)):
		print(pred[j])
		print(y_test[j],'\n')