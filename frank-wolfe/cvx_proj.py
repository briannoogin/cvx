import numpy as np

def main():
    # our cvx hull
    hull = np.array([[2,1], [4,3], [1,1.5], [.5,5]])
    
    # point we wish to project onto the set
    p_point = np.array([3,5])

    # "randomly" initialized point inside our convex set, set to the first vertex in our hull
    curr_proj = hull[0]

    epsilon = .01
    # if we want to arrive within epsilon of the correct answer, generally need 1/(epsilon^2) iter
    iter = int(1 / (epsilon ** 2)) 

    for z in range(iter):
        # step size required for convergence
        step = 2 / (z + 2)

        # gradient is 2*(z-y) where z is our current proj and y is the point we are attempting to proj
        gradient = 2*np.subtract(curr_proj, p_point)

        # calc s^T grad for each S vertex in our convex hull
        s_ks = [np.dot(vertex, gradient) for vertex in hull]

        # move towards the vertex which minimizes this
        s_k = np.argmin(s_ks)

        # update step
        curr_proj = curr_proj +  step * (np.subtract(hull[s_k], curr_proj))
    
    print(curr_proj)

if __name__ == "__main__":
    main()
