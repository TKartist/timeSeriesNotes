import numpy as np

ten_zeroes = np.zeros(19)
ten_ones = np.ones(10)
ten_fives = np.ones(10) * 5

ten_to_fifty = np.arange(10, 51)
t2f_even = ten_to_fifty[ten_to_fifty % 2 == 0]
print(t2f_even)

thr2thr = np.arange(0, 9).reshape(3, 3)
print(thr2thr)

identity_matrix = np.eye(3)
print(identity_matrix)

#random number between 0 and 1
print(np.random.rand(1))

# 25 random numbers standard normal distribution
print(np.random.randn(25))

# array of 20 linearly spaced points between 0 and 1
print(np.linspace(0, 1, 20))

print(np.arange(1, 26).reshape(5, 5)[2:, 1:])

print(np.arange(1, 26).sum())

# standard deviation
print(np.arange(1, 26).std())