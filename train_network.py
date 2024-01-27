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
# The data is shuffled before splitting into test and training sets
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

# data_test is the transpose of the first 1000 training examples, which will be used for testing the trained network
# Y_test is the first row of data_test, or an array of each of the ground-truth values of the first 1000 training examples
# X_test is the remaining 784 rows of data_test, each row representing the next pixle for each of the first 1000 training examples
# X_test values will be between 0 and 255, so X_test is divided by 255 to make each value between 0 and 1
data_test = data[0:1000].T
Y_test = data_test[0]
X_test = data_test[1:n]
X_test = X_test / 255.

# data_train is the transpose of the remaining 41,000 training examples, which will be used to train the network
# Y_train is the first row of data_train, or an array of each of the ground-truth values of the remaining 41,000 training examples
# X_train is the remaining 784 rows of data_train, each row representing the next pixle for each of the remaining 41,000 training examples
# X_train values will be between 0 and 255, so X_test is divided by 255 to make each value between 0 and 1
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

# An array of the number of nodes in each layer of the network
input = 784
output = 10
node_layers = [input, 10, output]

# np.random.rand(x, y) creates a matrix that looks like x arrays, y elements per array
# Each element is a random decimal between 0 and 1
# Each of the values of the matrices created in this method are subtracted by 0.5 to set them between -0.5 and 0.5
def init_params():
    params_array = []

    for i in range(len(node_layers)):
        if i != 0:
            biases = np.random.rand(node_layers[i], 1) - 0.5
            params_array.append(biases)
        if i != len(node_layers) - 1:
            weights = np.random.rand(node_layers[i + 1], node_layers[i]) - 0.5
            params_array.append(weights)

    return params_array

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
def forward_prop(params_array, X):
    computed_outputs = []

    for i in range(0, len(params_array), 2):
        # Z is the dot products of the previous layer and weights, plus the biases 
        if i == 0:
            Z = params_array[i].dot(X) + params_array[i + 1]
        else:
            Z = params_array[i].dot(computed_outputs[-1]) + params_array[i + 1]
        computed_outputs.append(Z)

        # A is Z times the activation function
        if i == len(params_array) - 2:
            A = softmax(Z)
        else:
            A = ReLU(Z)
        computed_outputs.append(A)

    return computed_outputs

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
def backward_prop(computed_outputs, params_array, X, Y):
    one_hot_Y = one_hot(Y)

    #############################

    # Z1 = computed_outputs[0]
    # A1 = computed_outputs[1]
    # Z2 = computed_outputs[2]
    # A2 = computed_outputs[3]

    # dZ2 = A2 - one_hot_Y
    # db2 = 1 / m * np.sum(dZ2)
    # dW2 = 1 / m * dZ2.dot(A1.T)

    # dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    # db1 = 1 / m * np.sum(dZ1)
    # dW1 = 1 / m * dZ1.dot(X.T)

    # loss_values = [dW1, db1, dW2, db2]

    #############################

    computed_outputs.reverse()

    loss_values = []
    for i in range(0, len(computed_outputs), 2):
        if i == 0:
            dZ = computed_outputs[0] - one_hot_Y
            db = 1 / m * np.sum(dZ)
            dW = 1 / m * dZ.dot(computed_outputs[i + 2].T)
        else:
            # dZ = params_array[i].T.dot(dZ) * ReLU_deriv(computed_outputs[i + 1])
            # db = 1 / m * np.sum(dZ)

            if i == len(computed_outputs) - 2:
                dZ = params_array[i].T.dot(dZ) * ReLU_deriv(computed_outputs[i + 1])
                db = 1 / m * np.sum(dZ)
                dW = 1 / m * dZ.dot(X.T)
            else:
                # dZ = params_array[i].T.dot(dZ) * ReLU_deriv(computed_outputs[i + 1])
                db = 1 / m * np.sum(dZ)
                dW = 1 / m * dZ.dot(computed_outputs[i + 2].T)

        loss_values.append(db)
        loss_values.append(dW)

    computed_outputs.reverse()
    loss_values.reverse()

    #############################

    return loss_values

def update_params(params_array, loss_values, alpha):
    new_params_array = []
    for i in range(len(params_array)):
        new_params_array.append(params_array[i] - alpha * loss_values[i])

    return new_params_array

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    params_array = init_params()

    for i in range(iterations):
        computed_outputs = forward_prop(params_array, X)
        loss_values = backward_prop(computed_outputs, params_array, X, Y)
        params_array = update_params(params_array, loss_values, alpha)

        print()
        print("Iteration:", i)
        predictions = get_predictions(computed_outputs[-1])
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

    return params_array

iterations = 500
params_array = gradient_descent(X_train, Y_train, 0.10, iterations)

# params_array = init_params()
# computed_outputs = forward_prop(params_array, X_train)
# loss_values = backward_prop(computed_outputs, params_array[-2], X_train, Y_train)

def make_predictions(X, params_array):
    computed_outputs = forward_prop(params_array, X)
    predictions = get_predictions(computed_outputs[-1])

    return predictions

test_predictions = make_predictions(X_test, params_array)

score_array = []
for i in range(test_predictions.size):
    if test_predictions[i] == Y_test[i]:
        score_array.append(test_predictions[i])
    else:
        score_array.append(str(test_predictions[i]) + "," + str(Y_test[i]))

print()
print("Test Set:")
print('[' + ', '.join(GREEN + str(score_array[i]) + RESET if score_array[i] == Y_test[i] else RED + str(score_array[i]) + RESET for i in range(len(score_array))) + ']')

print()
if get_accuracy(test_predictions, Y_test) < 0.5:
    print("Test Set Accuracy: " + RED + str(get_accuracy(test_predictions, Y_test)) + RESET)
elif get_accuracy(test_predictions, Y_test) < 0.8:
    print("Test Set Accuracy: " + YELLOW + str(get_accuracy(test_predictions, Y_test)) + RESET)
elif get_accuracy(test_predictions, Y_test) < 0.9:
    print("Test Set Accuracy: " + GREEN + str(get_accuracy(test_predictions, Y_test)) + RESET)
elif get_accuracy(test_predictions, Y_test) < 0.95:
    print("Test Set Accuracy: " + BLUE + str(get_accuracy(test_predictions, Y_test)) + RESET)
else:
    print("Test Set Accuracy: " + PURPLE + str(get_accuracy(test_predictions, Y_test)) + RESET)

np.set_printoptions(threshold=np.inf)

with open("weights_and_biases.txt", "w") as file:
    for i in range(len(params_array)):
        if i % 2 == 0:
            file.write("#W#")
        else:
            file.write("#b#")

        file.write(str(params_array[i]))
