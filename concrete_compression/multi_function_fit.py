import cvxpy as cvx  
import numpy as np
import os
import scipy
from load import load_data

def fit_cvx(train):
    # generate vars
    constraints = []
    obj = 0
    g_i = {}
    y_hat = {}

    for x in range(len(train)):
        # make g_i vector of len dim(input), else make it just a scalar if input is of dim = 1
        try:
            g_i["g_{0}".format(x)] = cvx.Variable(len(train[0][0]))
        except:
            g_i["g_{0}".format(x)] = cvx.Variable()
            
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
        obj += ((train[i][1] - y_hat["y_hat_{0}".format(i)]) ** 2)

    obj = cvx.Minimize(obj)

    # solve
    print('Solving...')
    prob = cvx.Problem(obj, constraints)
    prob.solve(verbose=True)

    g_hats = []
    y_hats = []

    # grab optimized y_hats and g_i's
    for i in range(len(g_i)):
        g_hats.append(g_i["g_{0}".format(i)].value)
        y_hats.append(y_hat["y_hat_{0}".format(i)].value)

    # note, actual answer is max{y_hat_1 + (g_1 * (x - x_train[0])), y_hat_2 + (g_2 * (x - x_train[1])), ...}

    return g_hats, y_hats, prob.solver_stats.solve_time

def test_fit(g_hats, y_hats, solve_time, x_test, y_test, train):
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

        for i in range(len(g_hats)):
            predictions.append(y_hats[i] + np.dot(g_hats[i], (x_i - train[i][0])))
            
        y_pred = np.amax(predictions)
        errors.append((y_pred - y_true) ** 2)
        print('{0:<25} {1}'.format(y_pred, y_true))
        all_preds.append(y_pred)
        all_true.append(y_true)

    print('Test batch size: ', len(x_test))
    print('Test mse: ', sum(errors) / len(errors))
    print('\nProb solve time: ', solve_time)

    return all_preds, all_true

def pred(x_test, y_test):
    print('Testing on test-set...')
    print('{0:25} {1}'.format('pred','real'))
    errors = []

    for p in range(len(x_test)):
        print('{0:<25} {1}'.format(x_test[p], y_test[p]))
        errors.append((x_test[p] - y_test[p]) ** 2)

    print('Testmse: ', sum(errors) / len(errors))

    return (sum(errors) / len(errors))
        
def main():    
    print('Load data of form...')
    # load_data already loads in as np arrays for us
    x_train, y_train, x_test, y_test = load_data()

    # number of train ex to fit
    train_batch = 10
    print('Fitting ', train_batch, '/', len(x_train))
    iterations = 17
    # train points
    train = [(x_train[i], y_train[i]) for i in range(train_batch)]
    nth_cvx_train = []
    nth_cvx_test = []

    for iter in range(iterations): 
        # fit two cvx functions, one to (x,y) and another to (x, f(x) - y).
        # fit first convex function
        g_hats, y_hats, solve_time = fit_cvx(train)
        #all_preds, _ = test_fit(g_hats, y_hats, solve_time, x_test, y_test, train)

        # first_cvx is array of (pred, real) tuples for the train batch
        cvx_train = []
        for i in range(train_batch):
            predictions = []
            for j in range(len(g_hats)):
                predictions.append(y_hats[j] + np.dot(g_hats[j], (train[i][0] - x_train[j])))

            y_pred = np.amax(predictions)
            cvx_train.append(y_pred)

        nth_cvx_train.append(cvx_train)

        # first cvx pred on x_test
        cvx_test = []
        for i in range(len(x_test)):
            predictions = []
            for j in range(len(g_hats)):
                predictions.append(y_hats[j] + np.dot(g_hats[j], (x_test[i] - x_train[j])))

            y_pred = np.amax(predictions)
            cvx_test.append(y_pred)

        nth_cvx_test.append(cvx_test)

        # store fit on the test_batch of f(x)
        print('train batch fit, cvx func {0}'.format(iter))
        print(cvx_train)

        # new training data is (-diff of cvx func and real, real)
        train = [(train[x][0], (cvx_train[x] - train[x][1])) for x in range(len(cvx_train))]
   
    print("First cvx fit")
    final_pred = nth_cvx_test[0]
    t_mse = []
    pred(final_pred, y_test)
    print("================================================")
    for i in range(1, len(nth_cvx_test)):
        final_pred = [(final_pred[j] - nth_cvx_test[i][j]) for j in range(len(nth_cvx_test[i]))]
        print("{0}'th cvx diff function".format(i))
        t_mse.append(pred(final_pred, y_test))
        print("================================================")
    print(t_mse) 
    #final_pred = [(x_test_first_cvx[i] - x_test_second_cvx[i]) for i in range(len(x_test_first_cvx))]

if __name__ == "__main__":
    main()
