import cvxpy as cvx  
import numpy as np
import os
import scipy
from load import load_data

print('Load data of form...')
# load_data already loads in as np arrays for us
x_train, y_train, x_test, y_test = load_data()

# number of train ex to fit
train_batch = 50
print('Fitting ', train_batch, '/', len(x_train))

# train points
train = [(x_train[i], y_train[i]) for i in range(train_batch)]

# generate vars
constraints = []
obj = 0
g_i = {}
y_hat = {}

for x in range(len(train)):
    g_i["g_{0}".format(x)] = cvx.Variable(8)
    y_hat["y_hat_{0}".format(x)] = cvx.Variable()

print('Setting up constraints...')
# straight outta boyd~
for i in range(len(train)):
    for j in range(len(train)):
        # leave out unnecessary constraint
        if i==j:
            continue
        constraints.append(y_hat["y_hat_{0}".format(i)] + g_i["g_{0}".format(i)].T * (train[j][0] - train[i][0]) <= y_hat["y_hat_{0}".format(j)])

print('Setting up objective...')
# set up obj to be sum of squared error
for i in range(len(y_hat)):
    obj += ((y_train[i] - y_hat["y_hat_{0}".format(i)]) ** 2)

print(obj)
obj = cvx.Minimize(obj)

# solve
print('Solving...')
prob = cvx.Problem(obj, constraints)
prob.solve(verbose=True)

sol = []
y_hats = []

# grab optimized y_hats and g_i's
for i in range(len(g_i)):
    sol.append(g_i["g_{0}".format(i)].value)
    y_hats.append(y_hat["y_hat_{0}".format(i)].value)

# note, actual answer is max{y_hat_1 + (g_1 * (x - x_train[0])), y_hat_2 + (g_2 * (x - x_train[1])), ...}

print('Testing on test-set...')
print('{0:25} {1}'.format('pred','real'))
# check if answer makes sense
for pred in range(len(x_test)):
    x_i = x_test[pred]
    y_true = y_test[pred]
    predictions = []

    for g_i in range(len(sol)):
        #print(train[g_i][0], sol[g_i])
        predictions.append(y_hats[g_i] + np.dot(sol[g_i], (x_i - x_train[g_i])))
        

    y_pred = np.amax(predictions)
    print('{0:<25} {1}'.format(y_pred, y_true))
print('\nProb solve time: ', prob.solver_stats.solve_time)

