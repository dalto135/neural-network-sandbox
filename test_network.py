import numpy as np
import pandas as pd
from random import random
import network_methods

data = pd.read_csv('train.csv')

# m is number of training examples (42,000)
# n is the number of pixels per example, plus 1 for the label column (28 x 28 + 1 = 785 in total)
# The data is shuffled before splitting into test and training sets
data = np.array(data)
m, n = data.shape
# np.random.shuffle(data)

# The transpose of data is taken to test the trained network
# Y is the first row of data, or an array of all of the ground-truth values of the data
# X is the remaining 784 rows of data, each row representing the next pixle of each of the images in data
# X values fall between 0 and 255, so it is divided by 255 to set each value between 0 and 1
data = data.T
Y = data[0]
# X = data[1:n]
X = data[1:1000]
X = X / 255.

file_path = "weights_and_biases.txt"
params_array = network_methods.get_params_from_file(file_path)
incorrect_guesses = network_methods.test_network_on_test_data(params_array, X, Y)

index = int(random() * len(incorrect_guesses))
index = incorrect_guesses[index]

network_methods.test_prediction(index, params_array, X, Y)
