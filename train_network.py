import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

RED = '\033[91m'
YELLOW = '\033[93m'
GREEN = '\033[92m'
BLUE = '\033[96m'
PURPLE = '\033[94m'
RESET = '\033[0m'

data = pd.read_csv('train.csv')

# m is number of training examples (42,000)
# n is the number of pixels per example, plus 1 for the label column (28 x 28 + 1 = 785 in total)
# Shuffle before splitting into dev and training sets
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

# data_dev is the transpose of the first 1000 training examples, which will be used for testing the trained network
# Y_dev is the first row of data_dev, or an array of each of the ground-truth values of the first 1000 training examples
# X_dev is the remaining 784 rows of data_dev, each row representing the next pixle for each of the first 1000 training examples
# X_dev values will be between 0 and 255, so X_dev is divided by 255 to make each value between 0 and 1
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

# data_train is the transpose of the remaining 41,000 training examples, which will be used to train the network
# Y_train is the first row of data_train, or an array of each of the ground-truth values of the remaining 41,000 training examples
# X_train is the remaining 784 rows of data_train, each row representing the next pixle for each of the remaining 41,000 training examples
# X_train values will be between 0 and 255, so X_dev is divided by 255 to make each value between 0 and 1
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

# np.random.rand(x, y) creates a matrix that looks like x arrays, y elements per array
# Each element is a random decimal between 0 and 1
# Ex. W1 and b1 correspond to layer 1, the hidden layer
# The hidden layer has 10 nodes, each recieving input from each of the 784 nodes of the input layer
# Each of the values of the matrices created in this method are subtracted by 0.5 to set them between -0.5 and 0.5
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2

# For RelU, Y = 0 when X <= 0, and Y = X when X > 0
def ReLU(Z):
    return np.maximum(Z, 0)

# This function is used instead of ReLU for the output layer of the network
# It sets each value as a percentage of confidence, all values adding up to 1
# np.exp(Z) is e^Z
# This function takes each element of the array and converts it to (e^element / the sum of e^each element)
def softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))

# X is a 784 x 41,000 matrix (X_train) each of the 784 arrays representing the next pixle for each of the 41,000 training examples
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    # 10x784 * 784x41,000 = 10x41,000
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    # 10x10 * 10x41,000 = 10x41,000
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

# For RelU, Y = 0 when X <= 0, and Y = X when X > 0
# Taking the derivative of this graph gives 0 when X <= 0 and 1 when X > 0
def ReLU_deriv(Z):
    return Z > 0

# Y is Y_train, an array of the 41,000 ground-truth numbers (labels) of the data used to train the network
# For each element in Y, this function outputs a 10 element array
# Each of these 10 elements is zero, except the element corresponding to the element from Y, which is set to 1
# Ex. If the ground-truth element is 3, the outputed 10 element array would be [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T

    return one_hot_Y

# Calculates the loss values for each of the weights and biases(?)
# Y is the ground-truth values
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)

    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)

    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)

    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i % 10 == 0:
            print()
            print("Iteration:", i)
            predictions = get_predictions(A2)
            if get_accuracy(predictions, Y) < 0.5:
                print("Accuracy: " + RED + str(get_accuracy(predictions, Y)) + RESET)
            elif get_accuracy(predictions, Y) < 0.8:
                print("Accuracy: " + YELLOW + str(get_accuracy(predictions, Y)) + RESET)
            elif get_accuracy(predictions, Y) < 0.9:
                print("Accuracy: " + GREEN + str(get_accuracy(predictions, Y)) + RESET)
            elif get_accuracy(predictions, Y) < 0.95:
                print("Accuracy: " + BLUE + str(get_accuracy(predictions, Y)) + RESET)
            else:
                print("Accuracy: " + PURPLE + str(get_accuracy(predictions, Y)) + RESET)

    return W1, b1, W2, b2

num_of_samples = 500
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, num_of_samples)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)

    return predictions

dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)

score_array = []
for i in range(dev_predictions.size):
    if dev_predictions[i] == Y_dev[i]:
        score_array.append(dev_predictions[i])
    else:
        score_array.append(str(dev_predictions[i]) + "," + str(Y_dev[i]))

print()
print("Test Set:")
print('[' + ', '.join(GREEN + str(score_array[i]) + RESET if score_array[i] == Y_dev[i] else RED + str(score_array[i]) + RESET for i in range(len(score_array))) + ']')

print()
if get_accuracy(dev_predictions, Y_dev) < 0.5:
    print("Test Set Accuracy: " + RED + str(get_accuracy(dev_predictions, Y_dev)) + RESET)
elif get_accuracy(dev_predictions, Y_dev) < 0.8:
    print("Test Set Accuracy: " + YELLOW + str(get_accuracy(dev_predictions, Y_dev)) + RESET)
elif get_accuracy(dev_predictions, Y_dev) < 0.9:
    print("Test Set Accuracy: " + GREEN + str(get_accuracy(dev_predictions, Y_dev)) + RESET)
elif get_accuracy(dev_predictions, Y_dev) < 0.95:
    print("Test Set Accuracy: " + BLUE + str(get_accuracy(dev_predictions, Y_dev)) + RESET)
else:
    print("Test Set Accuracy: " + PURPLE + str(get_accuracy(dev_predictions, Y_dev)) + RESET)

np.set_printoptions(threshold=np.inf)

with open("weights_and_biases.txt", "w") as file:
    file.write("W1#")
    file.write(str(W1))
    file.write("#b1#")
    file.write(str(b1))
    file.write("#W2#")
    file.write(str(W2))
    file.write("#b2#")
    file.write(str(b2))
