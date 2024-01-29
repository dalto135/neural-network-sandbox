import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

RED = '\033[91m'
YELLOW = '\033[93m'
GREEN = '\033[92m'
BLUE = '\033[96m'
PURPLE = '\033[94m'
RESET = '\033[0m'

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

# Calculates the loss values for each of the weights and biases(?)
def backward_prop(computed_outputs, params_array, one_hot_Y, X, m):
    loss_values = []

    for i in range(0, len(computed_outputs), 2):
        if i == 0:
            dZ = computed_outputs[-1] - one_hot_Y
        else:
            dZ = params_array[len(params_array) - i].T.dot(dZ) * ReLU_deriv(computed_outputs[len(computed_outputs) - i - 2])

        if i == len(computed_outputs) - 2:
            dW = 1 / m * dZ.dot(X.T)
        else:
            dW = 1 / m * dZ.dot(computed_outputs[len(computed_outputs) - i - 3].T)

        db = 1 / m * np.sum(dZ)

        loss_values.insert(0, db)
        loss_values.insert(0, dW)

    return loss_values

# Y is Y_train, an array of the 41,000 ground-truth numbers (labels) of the data used to train the network
# For each element in Y, this function outputs a 10 element array
# Each of these 10 elements is zero, except the element corresponding to the element from Y, which is set to 1
# Ex. If the ground-truth element is 3, the outputed 10 element array will be [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T

    return one_hot_Y

def update_params(params_array, loss_values, alpha):
    new_params_array = []
    for i in range(len(params_array)):
        new_params_array.append(params_array[i] - alpha * loss_values[i])

    return new_params_array

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# np.random.rand(x, y) creates a matrix that looks like x arrays, y elements per array
# Each element is a random decimal between 0 and 1
# Each of the values of the matrices created in this method are subtracted by 0.5 to set them between -0.5 and 0.5
def init_params(node_layers):
    params_array = []

    for i in range(len(node_layers)):
        if i != 0:
            biases = np.random.rand(node_layers[i], 1) - 0.5
            params_array.append(biases)
        if i != len(node_layers) - 1:
            weights = np.random.rand(node_layers[i + 1], node_layers[i]) - 0.5
            params_array.append(weights)

    return params_array

# Trains the network
def gradient_descent(X, Y, m, alpha, iterations, node_layers):
    params_array = init_params(node_layers)

    one_hot_Y = one_hot(Y)

    for i in range(iterations):
        computed_outputs = forward_prop(params_array, X)
        loss_values = backward_prop(computed_outputs, params_array, one_hot_Y, X, m)
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

    print()
    print("Training Complete!")

    return params_array

def make_predictions(X, params_array):
    computed_outputs = forward_prop(params_array, X)
    predictions = get_predictions(computed_outputs[-1])

    return predictions

def test_network_on_test_data(params_array, X_test, Y_test):
    test_predictions = make_predictions(X_test, params_array)

    score_array = []

    test_predictions = np.ravel(test_predictions)
    for i in range(len(test_predictions)):
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

def write_to_file(params_array):
    np.set_printoptions(threshold=np.inf)

    with open("weights_and_biases.txt", "w") as file:
        for i in range(len(params_array)):
            if i % 2 == 0:
                file.write("#W#")
            else:
                file.write("#b#")

            file.write(str(params_array[i]))

def get_params_from_file():
    with open("weights_and_biases.txt", "r") as file:
        weights_and_biases = file.read()

    array = weights_and_biases.split('#')

    params_array = []
    for i in range(0, len(array), 2):
        if i + 2 > len(array):
            break

        params = array[i + 2]
        params = params.replace("]\n [", "];\n [")
        params = np.matrix(params, dtype=float)
        params_array.append(params)

    return params_array

def test_prediction(index, params_array, X_test, Y_test):
    prediction = make_predictions(X_test[:, index, None], params_array)
    label = Y_test[index]
    print("Prediction:", prediction.item(0,0))
    print("Label:", label)

    if prediction != label:
        current_image = X_test[:, index, None]
        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show()
