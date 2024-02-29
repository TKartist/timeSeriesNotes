import numpy as np

mylist = [1, 2, 3]
# print(type(mylist))

arr = np.array(mylist) #-> converts it to numpy list

mylist = [[1, 2, 3],[4, 5, 6],[7, 8, 9]]
# print(mylist)
arr = np.array(mylist) # -> converts it to numpy list, basically matrix
# print(arr)

# [[1, 2, 3], [4, 5, 6], [7, 8, 9]] ->normal array
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]] -> matrix

ex = np.arange(0, 10) # -> np array of 0 to 9 np.arange(0, 11, 2) -> 0, 2, 4, 6, 8, 10
zs = np.zeros(5) # -> np array of 5 zeros
zs = np.zeros((4, 10)) # -> matrix of 4 by 10, 0s (ones works as well)

zs = np.ones((5, 5)) + 4

# [[5. 5. 5. 5. 5.]
#  [5. 5. 5. 5. 5.]
#  [5. 5. 5. 5. 5.]
#  [5. 5. 5. 5. 5.]
#  [5. 5. 5. 5. 5.]]
# numpy array can do addition, subtraction, division, and multiplication

zs = np.linspace(0, 10, 3) # -> [ 0.  5. 10.]
print(zs)

# np.eye(x) is identity matrix

zs = np.random.rand(4) # -> array of 4 random numbers (in case of randn instead of rand -> it is normally distributed with 0 as mean so (-1, 1) range)

zs = np.random.randint(1, 100, 10) # -> 10 random integer between range of [1, 100)

np.random.seed(555) # same seed = same random number
# print(np.random.rand(4))

arr = np.arange(25)
ranarr = np.random.randint(0, 50, 10)
arr = arr.reshape(5, 5) # -> goes from 1x25 to 5x5
# print(arr)
x = ranarr.argmax() # returns index of max val in array (max() returns max value), works same for min

#ranarr.dtype() -> data type

