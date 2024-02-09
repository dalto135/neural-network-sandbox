import numpy as np
import pandas as pd
import network_methods

data = pd.read_csv('train.csv')

# m is number of training examples (42,000)
# n is the number of pixels per example, plus 1 for the label column (28 x 28 + 1 = 785 in total)
# The data is shuffled before splitting into test and training sets
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

full_data = data.T
Y = full_data[0]
X = full_data[1:n]
X = X / 255.

# data_train is the transpose of the remaining 41,000 training examples, which will be used to train the network
# Y_train is the first row of data_train, or an array of each of the ground-truth values of the remaining 41,000 training examples
# X_train is the remaining 784 rows of data_train, each row representing the next pixle for each of the remaining 41,000 training examples
# X_train values fall between 0 and 255, so it is divided by 255 to set each value between 0 and 1
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

# data_test is the transpose of the first 1000 training examples, which will be used for testing the trained network
# Y_test is the first row of data_test, or an array of each of the ground-truth values of the first 1000 training examples
# X_test is the remaining 784 rows of data_test, each row representing the next pixle for each of the first 1000 training examples
# X_text values fall between 0 and 255, so it is divided by 255 to set each value between 0 and 1
data_test = data[0:1000].T
Y_test = data_test[0]
X_test = data_test[1:n]
X_test = X_test / 255.

# An array of the number of nodes in each layer of the network
input = 784
output = 10
node_layers = [input, 10, output]

iterations = 500
alpha = 0.10
params_array = network_methods.gradient_descent(X_train, Y_train, m, alpha, iterations, node_layers)
network_methods.test_network_on_test_data(params_array, X_test, Y_test)
# network_methods.test_network_on_test_data(params_array, X, Y)
network_methods.write_to_file(params_array)
