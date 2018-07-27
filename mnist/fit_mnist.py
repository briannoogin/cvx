import cvxpy as cvx  
import numpy as np
import os
import mnist
import scipy

def convert_and_save(vec, name):
    '''
    @Param: vec = [784,], name=save_dir
    :   Convert mnist vec to appropriate size, save as png

    '''
    im = np.reshape(vec, (28,28))
    #im = im.astype('uint16')
    scipy.misc.imsave(name, im)

def black_box(vec):
    '''
    @Param: x = [784, 1] flattened mnist
    :   Make a contiguous 7x7 black box in the image
    '''
    # create a contiguous (7x7) black square starting at 7th
    # row and 7th column of [28x28] mnist image
    for x in range(len(vec)):
        for col in range(7):
            for row in range(7,14):
                vec[x][(28*row) + 7 + col] = 0


# number of train ex to fit
train_batch = 6

# read in mnist data
x_train, _, _, _ = mnist.load()
x_train = x_train[:train_batch]

y_train, _, _, _ = mnist.load()
y_train = y_train[:train_batch]

x_test, _, _, _ = mnist.load()
x_test = x_test[-10:]

y_test, _, _, _ = mnist.load()
y_test = y_test[-10:]

black_box(x_train)
black_box(x_test)

# Training data --
#    x: squashed mnist row vector with 49 pixel contiguous piece missing
#    y: full original image

# test points
test = [(x_train[i], y_train[i]) for i in range(len(x_train))]

# generate vars
constraints = []
obj = 0
g_i = {}
y_hat = {}

for x in range(len(test)):
    g_i["g_{0}".format(x)] = cvx.Variable(784)
    y_hat["y_hat_{0}".format(x)] = cvx.Variable(784)

# straight outta boyd~
for i in range(len(test)):
    for j in range(len(test)):
        constraints.append(y_hat["y_hat_{0}".format(i)] + g_i["g_{0}".format(i)].T * (test[j][0] - test[i][0]) <= y_hat["y_hat_{0}".format(j)])

for j in range(len(y_hat)):
    constraints.append(y_hat["y_hat_{0}".format(j)] >= 0)

# set up obj to be sum of squared error
for i in range(len(test)):
    obj += cvx.sum_squares(test[i][1] - y_hat["y_hat_{0}".format(i)])
    #err = test[i][1] - y_hat["y_hat_{0}".format(i)]
    #err = [x ** 2 for x in err]
    #obj += sum(err)
obj = cvx.Minimize(obj)

# solve
prob = cvx.Problem(obj, constraints)
prob.solve(verbose=True)

sol = []
# print optimal linear values
for i in range(len(g_i)):
    #print("g_{0}".format(i),": ", g_i["g_{0}".format(i)].value)
    sol.append(g_i["g_{0}".format(i)].value)

# note, actual answer is max{g_1 * x, g_2 * x, ..., g_i * x}

test_index = 1
# check if answer makes sense
x_0 = x_test[test_index]
y_0 = y_test[test_index]
mse = []

for g_i in range(len(sol)):
    sol[g_i] = test[g_i][1] + sol[g_i] * (x_0 - test[g_i][0])
    sol[g_i] = [int(x) for x in sol[g_i]]
    err = sol[g_i] - y_0
    err = [x ** 2 for x in err]
    #print(err)
    squared_error = sum(err)
    mse.append(squared_error)

#y_hat = np.amax(sol)
print(mse)
min_index = mse.index(min(mse))
print('Minimum mse index: ', min_index)
print('Prob solve time: ', prob.solver_stats.solve_time)

convert_and_save(x_0, "images/orig_input.png")
#convert_and_save(np.invert(sol[min_index]), "images/inverted_cvx_ver_{0}.png".format(min_index))
convert_and_save(sol[min_index], "images/cvx_ver_{0}.png".format(min_index))
