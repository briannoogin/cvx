import numpy as np

def gradient_direction(hull, test_point):
    # return the direction of the gradient. This method always returns a point
    # from the hull, so instead we return the index of the point in hull to move towards. 
    gradient = [(np.subtract(vertex, test_point)) for vertex in hull]
    gradient = 2*sum(gradient)

    s_ks = [np.dot(hull[v], gradient) for v in range(len(hull))]

    # correct point to move towards is the minimum of all s_ks, which is a vertex from our constraint set
    return np.argmin(s_ks)

def main():
    # our cvx hull
    hull = np.array([[2,1], [4,3], [1,1.5], [.5,5]])
    
    # test point
    x = 1.2; y = 4;
    test_point = np.array([x,y])

    epsilon = .01
    # if we want to arrive within epsilon of the correct answer, generally need 1/(epsilon^2) iter
    #iter = int(1 / (epsilon ** 2)) 
    iter = 100

    for z in range(iter):
        step = 2 / (z + 2)
        s_k = gradient_direction(hull,test_point)
        test_point = test_point +  step * (np.subtract(hull[s_k], test_point))
    
    print(test_point)

if __name__ == "__main__":
    main()
