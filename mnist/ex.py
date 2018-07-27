import cvxpy as cvx  
import numpy as np
import os
import mnist

# read in mnist data
x_train, y_train_labels, x_test, y_test_labels = mnist.load()

'''
Training data --
    x: squashed mnist row vector with 5 pixel contiguous piece missing
    y: full original image

'''
x_train = x_train[:100]
y_train = x_train[:100]

#print(x_train)
#print(x_train.shape)

# test points
test = [(1,1), (2,2), (0,1), (1, 2), (3.5,3), (-4,4), (-8,4), (-2,0), (-6,2)]
#test = [(1,1), (2,2)]

# minimize piecewise linear function
def inference(pw_vars):
    '''
    @Params: pw_vars = [(a_i, b_i),...]
    pw_vars: convex piecewise-linear tuple set. Our actual function is max{pw_vars(x)}
    a_i, b_i = [n x 1] matrices (MNIST n = 784)
    
    returns:
    the input x that minimizes f(x) for the convex piecewise linear function
    '''
    x = cvx.Variable()
    t = cvx.Variable()
    constraints = []

    for i in range(len(pw_vars)):
        a_i, b_i = pw_vars[i]
        constraints.append(a_i*x + b_i <= t)

    obj = cvx.Minimize(t)
    prob = cvx.Problem(obj, constraints)
    prob.solve(verbose=True)
    
    print("inference status:", prob.status) 
    print("optimal value", prob.value) 
    print("optimal var", x.value)
    
    return x.value

# generate vars
constraints = []
obj = 0
g_i = {}
y_hat = {}

for x in range(len(test)):
    g_i["g_{0}".format(x)] = cvx.Variable()
    y_hat["y_hat_{0}".format(x)] = cvx.Variable()

# straight outta boyd
for i in range(len(test)):
    for j in range(len(test)):
        constraints.append(y_hat["y_hat_{0}".format(i)] + g_i["g_{0}".format(i)] * (test[j][0] - test[i][0]) <= y_hat["y_hat_{0}".format(j)])

# set up obj to be sum of squared error
for i in range(len(test)):
    obj += ((test[i][1] - y_hat["y_hat_{0}".format(i)]) ** 2)

obj = cvx.Minimize(obj)

# solve
prob = cvx.Problem(obj, constraints)
prob.solve(verbose=True)

sol = []
# print optimal linear values
for i in range(len(g_i)):
    print("g_{0}".format(i),": ", g_i["g_{0}".format(i)].value)
    sol.append(float(g_i["g_{0}".format(i)].value))

# debug --
x_0 = test[0][0]
y_0 = test[0][1]

for g_i in range(len(sol)):
    sol[g_i] = sol[g_i] * x_0

print(x_0, " ", np.amax(sol))
