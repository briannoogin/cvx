import numpy as np
import random
import time

'''
in_hull -- projection of point onto convex set
line_search -- convex combination of projection of point onto convex set
Both should work with arbitrary size inputs.
'''

def in_hull(hull, test_point):
    # given a covnex hull and a point, checks whether the point is witin the hull.
    # more specifically, checks whether the point within episilon of the projection 
    # of the point onto the hull.
    
    # "randomly" initialized point inside our convex set, set to a random vertex in the hull
    curr_proj = random.choice(hull)

    epsilon = .01
    # if we want to arrive within epsilon of the correct answer, generally need 1/(epsilon^2) iter
    iter = int(1 / (epsilon ** 2)) 

    for z in range(iter):
        # step size required for convergence
        step = 2 / (z + 2)

        # gradient is 2*(z-y) where z is our current proj and y is the point we are attempting to proj
        gradient = 2*np.subtract(curr_proj, test_point)

        # calc s^T grad for each S vertex in our convex hull
        s_ks = [np.dot(vertex, gradient) for vertex in hull]

        # move towards the vertex which minimizes this
        s_k = np.argmin(s_ks)

        # update step
        curr_proj = curr_proj +  step * (np.subtract(hull[s_k], curr_proj))
    
    print('proj:',curr_proj)
    # euclidean distance between projected and test_point
    distance = np.linalg.norm(curr_proj - test_point)  
    if distance > epsilon:
        print(distance, 'euclidean distance away')
        return False

    print('Within epsilon')
    return True
 
def line_search(hull, test_point):
    start = time.time()
    # returns: convex combination of projection of point onto a convex hull
    # uses line search + frank wolfe

    # stores all lambdas of the current convex combination of the curr_proj 
    cvx_comb = [0 for x in hull]

    # begin at a random vertex
    cvx_comb[random.randint(0, len(hull) - 1)] = 1

    epsilon = 0.01
    iter = int(1 / (epsilon ** 2)) 

    for z in range(1, iter):
        step = 2 / (z + 2)
        # current proj is current convex combination
        curr_proj = sum([cvx_comb[x] * hull[x] for x in range(len(hull))])
        
        # grad vector
        grad = 2*np.subtract(curr_proj, test_point)

        # s_k is the vertex we move towards
        s_k = np.argmin([np.dot(vertex, grad) for vertex in hull])
        # now find the projection of the test_point onto the line between curr_proj and s_k
        v = np.subtract(test_point, curr_proj)
        s = np.subtract(hull[s_k], curr_proj)
        # scalar_proj: our step size in the direction of the hull[s_k] - curr_proj vec
        scalar_proj = np.dot(v,s) / np.dot(s,s)

        # scale curr_proj by the projection of our testpoint onto the line joining curr_proj and s_k
        cvx_comb = [(1-scalar_proj) * x for x in cvx_comb]

        # add s_k to the convex combination
        cvx_comb[s_k] += scalar_proj

    end = time.time()
    print('Time:', end - start)

    proj_point = sum([cvx_comb[x] * hull[x] for x in range(len(hull))]) 
    # return a convex combination representing the projection, also return the projection itself
    return cvx_comb, proj_point

def main():
    # our cvx hull
    hull = np.array([[2,1], [4,3], [1,1.5], [.5,5]])
    
    # point we wish to project onto the set
    p_point = np.array([0,3])
    
    #_ = in_hull(hull, p_point)
    cvx_comb, proj = line_search(hull, p_point)
    print(proj)
    
if __name__ == "__main__":
    main()
