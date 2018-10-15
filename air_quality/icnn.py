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
# after some searching, these seem to be optimal values
iter_ = 1000
lr = 1e-2
batch_size = 32
fc1_size = 200
fc2_size = 200
nGradIter = 5 # from icnn paper

# number of variables we consider in each x input
num_vars = 14

init = tf.initialize_all_variables()

# XY defines input space. Y will be optimized over in the training cycle
X = tf.placeholder(tf.float32, [None, num_vars])
Y = tf.placeholder(tf.float32, [None, 1])

# the real Y value passed to us
Y_true = tf.placeholder(tf.float32, [None, 1])

xy = tf.concat([X, Y], 1)
num_vars += 1

# fc1
W_1 = tf.Variable(tf.truncated_normal([num_vars, fc1_size]))
b_1 = tf.Variable(tf.truncated_normal([fc1_size]))
fc1 = tf.nn.softmax(tf.matmul(xy, W_1) + b_1)

# fc2
W_2 = tf.Variable(tf.truncated_normal([fc1_size,fc2_size]))
b_2 = tf.Variable(tf.truncated_normal([fc2_size]))
fc2 = tf.nn.softmax(tf.matmul(fc1, W_2) + b_2)

#fc3 (out)
W_3 = tf.Variable(tf.truncated_normal([fc2_size,1]))
b_3 = tf.Variable(tf.truncated_normal([1]))
energy = tf.matmul(fc2, W_2) + b_2

# clipping op so weights stay in [0,inf)
#clip_W_1 = W_1.assign(tf.maximum(0., W_1)) # First weights don't need to be clipped
clip_W_2 = W_2.assign(tf.maximum(0., W_2))
clip_W_3 = W_3.assign(tf.maximum(0., W_3)) 
clip = tf.group(clip_W_2, clip_W_3)

# grad_op for computing minimum input Y
grad_op = tf.gradients(energy, Y)[0]

# train_op for when we actually modify the network (ie once we've found the minimum)
cost = tf.reduce_mean(tf.square(Y - Y_true))
train_op = tf.train.GradientDescentOptimizer(lr).minimize(energy)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    train_path = 'air_quality_train'
    val_path = 'air_quality_test'

    print('Data snippit...')
    # load same data as cvx fit
    x_train, y_train = load_data('air_quality_train')
    x_test, y_test = load_data('air_quality_test')

    # normalize input data
    x_train = sklearn.preprocessing.scale(x_train)
    x_test = sklearn.preprocessing.scale(x_test)

    # convert y_train to tf usable shape (not (148,) np shape)
    y_train = np.reshape(y_train, [len(y_train),1])
 
    # training cycle
    for i in range(iter_):
        
        # sklearn shuffle to get minibatch
        shuffled_x, shuffled_y = sklearn.utils.shuffle(x_train, y_train, n_samples=batch_size)
        
        # compute minimums of the energy function
        y_i = np.zeros(batch_size)
        v_i = np.zeros(batch_size)
        momentum = 0.9
        lr = 0.1
        for j in range(nGradIter):
            prev_v_i = v_i 
            grads = sess.run([grad_op], feed_dict={X: shuffled_x, Y: y_i, Y_true: shuffled_y})
            v_i = momentum*prev_v_i - lr*grads
            y_i = y_i - momentum*prev_v_i + (1+momentum)*v_i

        # 1 iteration of training by feeding in the computed minimum as Y
        _, current_cost = sess.run([train, cost], feed_dict={X: shuffled_x, Y_: y_i, Y_true: shuffled_y}) 
        # clip weights to maintain convexity
        sess.run(clip)

        # print progress
        if i % 100 == 0:
            print('Iter: {0} Train mse: {1}'.format(i, current_cost))

    print('\nTesting on test-set...')
    pred = sess.run(out, feed_dict={X: x_test})
    error = []

    # test on val set
    print('{0:25} {1}'.format('pred', 'real'))
    for j in range(len(pred)):
        #print('{0:<25} {1}'.format(str(pred[j][0]), y_test[j]))
        error.append((y_test[j] - pred[j][0]) ** 2)

    print('\nTest batch size: ', len(y_test))
    print('Test mse: ', sum(error) / len(error))
    
    '''
    ## check if weight tensors are all non-negative
    tvars = tf.trainable_variables()
    tvars_vals = sess.run(tvars)
    for var, val in zip(tvars, tvars_vals):
        print(var.name, val)
    '''
