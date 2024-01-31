import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

RED = '\033[91m'
YELLOW = '\033[93m'
GREEN = '\033[92m'
BLUE = '\033[96m'
PURPLE = '\033[94m'
RESET = '\033[0m'

# For RelU, Y = 0 when X <= 0, and Y = X when X > 0.
def ReLU(Z):
    return np.maximum(Z, 0)

# This function is used instead of ReLU for the output layer of the network.
# It sets each value as a percentage of confidence, all values adding up to 1.
# np.exp(Z) is e^Z
# This function takes each element of the array and converts it to (e^element / the sum of e^each element).
def softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))

# Returns the computed outputs for each layer of nodes in the network,
# before and after performing the activation function, for each training example in X.
# X is a 784 x 41,000 matrix (X_train) each of the 784 arrays representing the next pixle for each of the 41,000 training examples.
def forward_prop(params_array, X):
    computed_outputs = []

    for i in range(0, len(params_array), 2):
        # Z is the dot products of the previous layer and weights, plus the biases.
        if i == 0:
            Z = params_array[i].dot(X) + params_array[i + 1]
        else:
            Z = params_array[i].dot(computed_outputs[-1]) + params_array[i + 1]
        computed_outputs.append(Z)

        # A is Z times the activation function.
        if i == len(params_array) - 2:
            A = softmax(Z)
        else:
            A = ReLU(Z)
        computed_outputs.append(A)

    return computed_outputs

# For RelU, Y = 0 when X <= 0, and Y = X when X > 0.
# Taking the derivative of this graph gives 0 when X <= 0 and 1 when X > 0.
# Therefore, returns either a 0 or 1 for each element in Z.
def ReLU_deriv(Z):
    return Z > 0

# Calculates the loss values for each of the weights and biases of the network,
# based on the outputs it gave when presented the training data.
# The diagram below is an example of the loss values calculation for a network with two hidden layers,
# when the loop is on the second hidden layer.

#                                              Current layer
#                                                   \/
#                                    X      Z1      Z2      Z3
#                                    X      A1      A2      A3
#                                    X      dZ1     dZ2     dZ3
#                                    X  dW1 db1 dW2 db2 dW3 db3
#                                    X  W1  b1  W2  b2  W3  b3

#                               computed_values = [Z1, A1, Z2, A2, Z3, A3]
#                               params_array = [W1, b1, W2, b2, W3, b3]
#                               i = 2
#                               dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2)
#                               db2 = 1 / m * np.sum(dZ2)
#                               dW2 = 1 / m * dZ2.dot(A1.T)
def backward_prop(computed_outputs, params_array, one_hot_Y, X, m):
    loss_values = []

    # This loop iterates backwards through each biases layer (layer of nodes) in the network.
    for i in range(0, len(computed_outputs), 2):
        # dZ is the loss values that are calculated for the biases of the current layer, for each of the training examples.
        if i == 0:
            # If the current layer is the output layer, 
            # the loss values are the computed outputs minus the ground-truth outputs.
            dZ = computed_outputs[-1] - one_hot_Y
        else:
            # Else, dZ is calculated by taking the dot product of the weights layer to the right of the current layer,
            # and the loss values of the biases layer to the right of the current layer.
            # This is then multiplied by ReLU_deriv() of the computed outputs of the current layer,
            # before the activation function was performed on these computed outputs.
            # ReLU_deriv() returns either a 0 or 1 for each training example.
            dZ = params_array[len(params_array) - i].T.dot(dZ) * ReLU_deriv(computed_outputs[len(computed_outputs) - i - 2])

        # Once dZ is calculated, it is used to calculate the loss values of the current biases layer,
        # and the loss values of the weights layer to the left of the current layer, per training example.
        # The loss values are added together and divided by the number of training examples (m),
        # which gives the average loss value for each parameter in these two layers.

        # db is the averaged loss values of the biases of the current layer.
        db = 1 / m * np.sum(dZ)

        # dW is the averaged loss values for the weights layer to the left of the current layer.
        if i == len(computed_outputs) - 2:
            # If the current layer is second after the input layer,
            # dW is the dot product of the loss values of the current layer and the values of the input layer.
            # The input layer has no bias values, the input values are taken directly.
            dW = 1 / m * dZ.dot(X.T)
        else:
            # Else, dW is the dot product of the loss values of the current layer,
            # and the computed outputs of the biases layer to the left of the current layer,
            # after the activation function was performed on these computed outputs.
            dW = 1 / m * dZ.dot(computed_outputs[len(computed_outputs) - i - 3].T)

        # Each element is inserted at the beginning of the array since the iteration travels backwards.
        loss_values.insert(0, db)
        loss_values.insert(0, dW)

    return loss_values

# Y is Y_train, an array of the 41,000 ground-truth numbers (labels) of the data used to train the network.
# For each element in Y, this function outputs a 10 element array.
# Each of these 10 elements is zero, except the element corresponding to the element from Y, which is set to 1.
# Ex. If the ground-truth element is 3, the outputed 10 element array will be [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T

    return one_hot_Y

# Updates the weights and biases of the network by taking each parameter,
# and subtracting the loss value for that parameter times some value alpha.
def update_params(params_array, loss_values, alpha):
    new_params_array = []
    for i in range(len(params_array)):
        new_params_array.append(params_array[i] - alpha * loss_values[i])

    return new_params_array

# Takes the outputs of the final layer of the network,
# and returns the index of the number assigned the highest probability, for each training example.
def get_predictions(A2):
    return np.argmax(A2, 0)

# Calculates the accuracy of the network by comparing the number of correct guesses,
# to the total number of examples in the training data.
def get_accuracy(predictions, Y):
    num_correct = np.sum(predictions == Y)
    percentage = num_correct / Y.size
    accuracy_string = "\n" + str(num_correct) + " / " + str(Y.size) + "\n" + str(percentage)

    if percentage < 0.5:
        accuracy_string = RED + accuracy_string + RESET
    elif percentage < 0.8:
        accuracy_string = YELLOW + accuracy_string + RESET
    elif percentage < 0.9:
        accuracy_string = GREEN + accuracy_string + RESET
    elif percentage < 0.95:
        accuracy_string = BLUE + accuracy_string + RESET
    else:
        accuracy_string = PURPLE + accuracy_string + RESET

    return accuracy_string

# np.random.rand(x, y) creates a matrix that looks like x arrays, y elements per array.
# Each element is a random decimal between 0 and 1, which is then subtracted by 0.5 to set them between -0.5 and 0.5.
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

# Trains the network given:
# X: The images used for training
# Y: The ground-truth numbers for X
# alpha: Value used for adjusting parameters with loss values after backpropagation
# iterations: Number of training iterations
# node_layers: Array of the number of nodes in each layer of the network
def gradient_descent(X, Y, m, alpha, iterations, node_layers):
    params_array = init_params(node_layers)

    one_hot_Y = one_hot(Y)

    for i in range(iterations):
        computed_outputs = forward_prop(params_array, X)
        loss_values = backward_prop(computed_outputs, params_array, one_hot_Y, X, m)
        params_array = update_params(params_array, loss_values, alpha)

        predictions = get_predictions(computed_outputs[-1])
        accuracy_string = get_accuracy(predictions, Y)

        print()
        print("Iteration:", i)
        print("Training Set Accuracy: " + accuracy_string)

    print()
    print("Training Complete!")

    return params_array

def make_predictions(X, params_array):
    computed_outputs = forward_prop(params_array, X)
    predictions = get_predictions(computed_outputs[-1])

    return predictions

def test_network_on_test_data(params_array, X_test, Y_test):
    test_predictions = make_predictions(X_test, params_array)
    test_predictions = np.ravel(test_predictions)

    score_array = []
    for i in range(len(test_predictions)):
        if test_predictions[i] == Y_test[i]:
            score_array.append(test_predictions[i])
        else:
            score_array.append(str(test_predictions[i]) + "," + str(Y_test[i]))

    print()
    print("Test Set:")
    print('[' + ', '.join(GREEN + str(score_array[i]) + RESET if score_array[i] == Y_test[i] else RED + str(score_array[i]) + RESET for i in range(len(score_array))) + ']')

    accuracy_string = get_accuracy(test_predictions, Y_test)

    print()
    print("Test Set Accuracy: " + accuracy_string)

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
