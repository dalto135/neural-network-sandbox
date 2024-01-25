import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

with open("weights_and_biases.txt", "r") as file:
    weights_and_biases = file.read()

array = weights_and_biases.split('#')

W1 = array[1]
# print(W1)
# W1 = np.matrix(W1, dtype=int)
b1 = array[3]
print(b1)
# b1 = np.matrix(b1, dtype=int)
W2 = array[5]
print(W2)
# W2 = np.matrix(W2, dtype=int)
b2 = array[7]
print(b2)
# b2 = np.matrix(b2, dtype=int)



def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

def get_predictions(A2):
    return np.argmax(A2, 0)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)

    return predictions

def test_prediction(index, W1, b1, W2, b2):
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction:", prediction)
    print("Label:", label)
    
    # current_image = X_train[:, index, None]
    # current_image = current_image.reshape((28, 28)) * 255
    # plt.gray()
    # plt.imshow(current_image, interpolation='nearest')
    # plt.show()

# test_prediction(0, W1, b1, W2, b2)
# test_prediction(1, W1, b1, W2, b2)
# test_prediction(2, W1, b1, W2, b2)
# test_prediction(3, W1, b1, W2, b2)