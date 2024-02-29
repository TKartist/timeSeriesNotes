import numpy as np

arr = np.arange(0, 11)
print(arr[1:4]) # -> prints index 1 to 4 (excluding last)
print(arr[:5]) # 0 to 5
print(arr[1:]) # from 1st (including) index
print(arr ** 2) # squares all value in arr
print(arr * 2)

slice_arr = arr[0:6]
# slice_arr[:] = 99
print(slice_arr)
print(arr)
# slice_arr are still pointing to arr so the arr values get changed too
# to avoid this:
arr_copy = arr.copy()
arr_copy[:] = 99
print(arr_copy)
print(arr)

# indexing 2D array
arr_2d = np.array([[5, 10, 15, 1], [20, 25, 30, 2], [35, 40, 45, 3]])
print(arr_2d.shape) # -> 3x3
print(arr_2d[:2, 1:])

arr = np.arange(1, 11)
print(arr > 4) # boolean array of condition satisfied or not

# clever thing to do:
bool_arr = arr > 4
print(arr[bool_arr]) # prints only elements which satisfies the condition

# you can do arr + arr as well

# numpy, you don't throw error, it will hold warnings with values like NaN and inf
# numpy can do math operations like np.log or np.sin etc

print(arr_2d)
print(arr_2d.sum(axis=0)) # col sum
print(arr_2d.sum(axis=1)) # row sum