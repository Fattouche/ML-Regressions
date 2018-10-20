import numpy as np
import matplotlib.pyplot as plt

filename = "regdata.csv"

'''
E = (1/(2*n)) * ( np.sum((y-w@X.T)**2) )
print("E = ", E)
w = w + kappa*( (1/n)*( np.sum( (y-w@X.T).T*X, axis=0, keepdims=True ) ) )
print("new w = ", w)
E = (1/(2*n)) * ( np.sum((y-w@X.T)**2) )
'''


def read_file():
    file = open(filename, "r")
    temp = []
    for index, line in enumerate(file):
        arr = line.rstrip().split(",")
        arr.insert(0, 1)
        temp.append(list(map(float, arr)))
    return np.array(temp)


def scale_attributes(data):
    attr_max = np.amax(data, axis=0)
    attr_min = np.amin(data, axis=0)
    attr_avg = np.mean(data, axis=0)
    for Xi_index in range(len(data)):
        for index in range(1, len(data[0])):
            data[Xi_index][index] = (
                data[Xi_index][index] - attr_avg[index])/(attr_max[index]-attr_min[index])
    return data


def format_x_y(data):
    x = data[:, [0, 1, 2]]
    y = data[:, [3]]
    return x, y


def main():
    kappa = 0.5
    data = read_file()
    data = scale_attributes(data)
    X, y = format_x_y(data)
    n = len(X)
    w = np.array([0, 0, 0])
    error = []
    for i in range(100):
        error.append((1/(2*n)) * (np.sum((y.T-w@X.T)**2)))
        w = w + kappa*((1/n)*(np.sum((y.T-w@X.T).T*X, axis=0, keepdims=True)))

    plt.plot([i for i in range(100)], error)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Error at Each Training Iteration')
    plt.show()


main()
