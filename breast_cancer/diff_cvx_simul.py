import cvxpy as cvx  
import numpy as np
import os
import scipy
from load import load_data

def fit_cvx(train):
    # generate vars
    constraints = []
    obj = 0
    g_i_1 = {}
    y_hat_1 = {}
    g_i_2 = {}
    y_hat_2 = {}

    for x in range(len(train)):
        # make g_i vector of len dim(input), else make it just a scalar if input is of dim = 1
        try:
            g_i_1["g_{0}".format(x)] = cvx.Variable(len(train[0][0]))
            g_i_2["g_{0}".format(x)] = cvx.Variable(len(train[0][0]))
        except:
            g_i_1["g_{0}".format(x)] = cvx.Variable()
            g_i_2["g_{0}".format(x)] = cvx.Variable()
 
        y_hat_1["y_hat_{0}".format(x)] = cvx.Variable()
        y_hat_2["y_hat_{0}".format(x)] = cvx.Variable()
 
    print('Setting up constraints...')
    # straight outta boyd~
    for i in range(len(train)):
        for j in range(len(train)):
            # leave out unnecessary constraint
            if i==j:
                continue
            constraints.append(y_hat_1["y_hat_{0}".format(i)] + g_i_1["g_{0}".format(i)].T * (train[j][0] - train[i][0]) <= y_hat_1["y_hat_{0}".format(j)])
            constraints.append(y_hat_2["y_hat_{0}".format(i)] + g_i_2["g_{0}".format(i)].T * (train[j][0] - train[i][0]) <= y_hat_2["y_hat_{0}".format(j)])

 
    print('Setting up objective...')
    # set up obj to be sum of squared error
    for i in range(len(y_hat_1)):
        obj += ((train[i][1] - (y_hat_1["y_hat_{0}".format(i)] - y_hat_2["y_hat_{0}".format(i)])) ** 2)

    obj = cvx.Minimize(obj)

    # solve
    print('Solving...')
    prob = cvx.Problem(obj, constraints)
    prob.solve(verbose=True)

    g_hats_1 = []
    y_hats_1 = []
    g_hats_2 = []
    y_hats_2 = []

    # grab optimized y_hats and g_i's
    for i in range(len(g_i_1)):
        g_hats_1.append(g_i_1["g_{0}".format(i)].value)
        y_hats_1.append(y_hat_1["y_hat_{0}".format(i)].value)
        g_hats_2.append(g_i_2["g_{0}".format(i)].value)
        y_hats_2.append(y_hat_2["y_hat_{0}".format(i)].value)


    # note, actual answer is max{y_hat_1 + (g_1 * (x - x_train[0])), y_hat_2 + (g_2 * (x - x_train[1])), ...}

    return g_hats_1, y_hats_1, g_hats_2, y_hats_2, prob.solver_stats.solve_time

def test_fit(g_hats_1, y_hats_1, g_hats_2, y_hats_2, solve_time, x_test, y_test, train):
    errors = []
    all_preds = []
    all_true = []

    print('Testing cvx1 on test-set...')
    print('{0:25} {1}'.format('pred','real'))
    # check if answer makes sense
    for pred in range(len(x_test)):
        x_i = x_test[pred]
        y_true = y_test[pred]
        predictions = []

        for i in range(len(g_hats_1)):
            y_1 = y_hats_1[i] + np.dot(g_hats_1[i], (x_i - train[i][0]))
            y_2 = y_hats_2[i] + np.dot(g_hats_2[i], (x_i - train[i][0]))
            predictions.append(y_1 - y_2)

        y_pred = np.amax(predictions)
        errors.append((y_pred - y_true) ** 2)
        print('{0:<25} {1}'.format(y_pred, y_true))
        all_preds.append(y_pred)
        all_true.append(y_true)

    print('Test batch size: ', len(x_test))
    print('Test mse: ', sum(errors) / len(errors))
    print('\nProb solve time: ', solve_time)

    return all_preds, all_true


def main():    
    print('Load data of form...')
    # load_data already loads in as np arrays for us
    x_train, y_train, x_test, y_test = load_data()

    # number of train ex to fit
    train_batch = 15
    print('Fitting ', train_batch, '/', len(x_train))

    # train points
    train = [(x_train[i], y_train[i]) for i in range(train_batch)]

    # fit first convex function
    g_hats_1, y_hats_1, g_hats_2, y_hats_2, solve_time = fit_cvx(train)
    all_preds, y_test = test_fit(g_hats_1, y_hats_1, g_hats_2, y_hats_2, solve_time, x_test, y_test, train)

if __name__ == "__main__":
    main()
