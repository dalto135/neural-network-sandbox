import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_test = data[0:1000].T
Y_test = data_test[0]
X_test = data_test[1:n]
X_test = X_test / 255.

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

def ReLU(Z):
    return np.maximum(Z, 0)

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
        if i + 1 == len(params_array) - 1:
            A = softmax(Z)
        else:
            A = ReLU(Z)
        computed_outputs.append(A)

    return computed_outputs

def get_predictions(A2):
    return np.argmax(A2, 0)

def make_predictions(X, params_array):
    computed_outputs = forward_prop(params_array, X)
    predictions = get_predictions(computed_outputs[-1])

    return predictions

def test_prediction(index, params_array):
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

test_prediction(0, params_array)
test_prediction(1, params_array)
test_prediction(2, params_array)
test_prediction(3, params_array)
