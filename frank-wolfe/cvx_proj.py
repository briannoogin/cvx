import numpy as np
import random
import time

'''
in_hull -- projection of point onto convex set
line_search -- convex combination of projection of point onto convex set
binary_search -- calls both above functions to find intersection of line with convex set
and return it as a convex combination of the vertices in the convex hull.

Notes:
1. all 'should' work with arbitrary size inputs.
2. Maybe use cython to speed it up? would have to reimplement some stuff though because np arrays
are inefficient to call in cython.
'''

def in_hull(hull, test_point):
    # given a convex hull and a point, computes the projection of the point onto the hull. If the point
    # is already in the hull, returns that point.
    
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
    
    return curr_proj
 
def line_search(hull, test_point):
    start = time.time()
    # returns: convex combination of projection of point onto a convex hull, as well as the projection
    # itself

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
    print('Projection Time:', end - start)

    proj_point = sum([cvx_comb[x] * hull[x] for x in range(len(hull))]) 
    # return a convex combination representing the projection, also return the projection itself
    return cvx_comb, proj_point

def binary_search(hull, point):
    # search for the x_n \in [x_1, x_2, ... x_n-1] that is equal to it's projection. 
    # searching alone the entire line using binary search.
    epsilon = 0.1

    start = time.time()
    # find a suitable y value to start our binary search. This value is the max y value of the hull
    all_ys = [hull[x][len(hull[x]) - 1] for x in range(len(hull))]
    max_y = max(all_ys)
    min_y = min(all_ys)
    
    # our projected point must be min_y < y < max_y
    test_point = np.append(point, max_y)
   
    # see if our inital guess is actually it's own projection onto the hull
    proj_found = in_hull(hull, test_point)

    # distance from the binary searched y to the projected y
    distance = test_point[len(test_point) - 1] - proj_found[len(proj_found) - 1]
    print(distance)
    prev_max_y = max_y
    print(test_point) 

    # binary search
    for x in range(10):
        if distance < epsilon:
            # we are now inside the hull, need to move back outside
            # update the min_y and revert to prevex max_y to move back
            min_y = max_y
            max_y = prev_max_y

        # outside the hull, so we move to halfway between our min and max
        prev_max_y = max_y
        print('prev max y', prev_max_y)
        max_y = (max_y + min_y) / 2
        test_point[len(test_point) - 1] = max_y
        print(test_point) 

        # recompute projection and distance
        proj_found = in_hull(hull, test_point)
        print('proj found', proj_found)
        distance = test_point[len(test_point) - 1] - proj_found[len(proj_found) - 1]
        print('new distance', distance)
    
    # now, find the point as a convex combination of the vertices
    proj = line_search(hull, test_point)
    end = time.time()
    print('Binary search + line_search time: ', end - start)

    # proj is a tuple (convex combination, projection)
    return proj

def main():
    # our cvx hull (example, needs to be in np.array)
    hull = np.array([[2,1], [4,3], [1,1.5], [.5,5]])
    
    # point we wish to project onto the set
    p_point = np.array([3,6])  
    cvx_comb, proj = line_search(hull, p_point)
   
    # now use binary search to look for the projection of the point along the line x=2
    # this should return the same thing as proj((2,5)) onto the convex set.
    
    # line to search along 
    x = 1.4 
    proj_point = binary_search(hull, np.array(x))
    
    print("\nBinary searched projection along line x={0}: ".format(x), proj_point[1])
    print("Point as a convex combination of hull vertices: ", proj_point[0])

if __name__ == "__main__":
    main()
