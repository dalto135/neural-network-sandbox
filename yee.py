import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



# W2 = np.random.rand(10, 5)
# print(W2)
# print(W2.size)
# print(len(W2))
# print(len(W2[0]))
# print(W2[0].size)

# array = [
#     [5, 5, 5, 5, 5],
#     [5, 5, 5, 5, 5],
#     [5, 5, 5, 5, 5],
#     [5, 5, 5, 5, 5],
#     [5, 5, 5, 5, 5],
#     [5, 5, 5, 5, 5],
#     [5, 5, 5, 5, 5],
#     [5, 5, 5, 5, 5],
#     [5, 5, 5, 5, 5],
#     [5, 5, 5, 5, 5]
# ]

# print(len(array))
# print(len(array[0]))

matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[1, 2], [3, 4]])

# 1  3
# 2  4
twod_answer = matrix1.dot(matrix2)
print(twod_answer)

twod_reshape = twod_answer.T
print(twod_reshape)

matrix3 = np.matrix('1 2 3; 4 5 6; 7 8 9')
matrix4 = np.matrix('1 2 3; 4 5 6; 7 8 9')

print(matrix3.dot(matrix4))