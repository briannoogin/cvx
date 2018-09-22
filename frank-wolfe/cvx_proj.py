import numpy as np

def n_distance(a, b):
    d = 0
    for x in range(len(a)):
        d += a[x] - b[x]
    
    return d

def gradient_direction(hull, test_point):
    # return the direction of the gradient. This method always returns a point
    # from the hull, so instead we return the index of the point in hull to move towards.
    gradient = [2*(n_distance(hull[x], test_point)) for x in range(len(hull))]

    # correct point to move towards is the minimum index of gradient
    direction = gradient.index(min(gradient))
    #print(gradient)

    return direction

def main():
    # our cvx hull
    hull = np.array([[-4,2], [.5,-2.5], [6,3]])

    # x coordinate of our test point. In this case, the corresponding y value should be 4
    x = -6; y = 0;
    test_point = np.array([x,y])

    # how close to get to correct answer
    epsilon = .01
    # if we want to arrive within epsilon of the correct answer, generally need 1/(epsilon^2) iter
    iter = int(1 / (epsilon ** 2))

    for z in range(1, iter):
        step = 2 / (z + 2)
        min_gradient_index = gradient_direction(hull,test_point)
        print(np.subtract(hull[min_gradient_index], test_point)) 
        test_point = np.add(test_point, step * (np.subtract(hull[min_gradient_index], test_point)))
        #print(z,'/',iter)
    
    print('======')
    print(test_point)

if __name__ == "__main__":
    main()
