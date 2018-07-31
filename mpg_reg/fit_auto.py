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
    g_i["g_{0}".format(x)] = cvx.Variable(4)
    y_hat["y_hat_{0}".format(x)] = cvx.Variable()

print('Setting up constraints...')
# straight outta boyd~
for i in range(len(train)):
    for j in range(len(train)):
        constraints.append(y_hat["y_hat_{0}".format(i)] + g_i["g_{0}".format(i)].T * (train[j][0] - train[i][0]) <= y_hat["y_hat_{0}".format(j)])

for j in range(len(y_hat)):
    constraints.append(y_hat["y_hat_{0}".format(j)] >= 0)

print('Setting up objective...')
# set up obj to be sum of squared error
for i in range(len(train)):
    obj += ((train[i][1] - y_hat["y_hat_{0}".format(i)]) ** 2)

obj = cvx.Minimize(obj)

# solve
print('Solving...')
prob = cvx.Problem(obj, constraints)
prob.solve(verbose=True)

sol = []
# print optimal linear values
for i in range(len(g_i)):
    #print("g_{0}".format(i),": ", g_i["g_{0}".format(i)].value)
    sol.append(g_i["g_{0}".format(i)].value)

# note, actual answer is max{g_1 * x, g_2 * x, ..., g_i * x}

print('Testing on test-set...')
print('{0:25} {1}'.format('pred','real'))
# check if answer makes sense
for pred in range(train_batch):
    x_i = x_train[pred]
    y_true = y_train[pred]

    for g_i in range(len(sol)):
        #print(np.multiply(sol[g_i], (x_i - train[g_i][0])))
        sol[g_i] = train[g_i][1] + np.multiply(sol[g_i], (x_i - train[g_i][0]))

    y_pred = np.amax(sol)
    print('{0:<25} {1}'.format(y_pred, y_true))

print('\nProb solve time: ', prob.solver_stats.solve_time)

