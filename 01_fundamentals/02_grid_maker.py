import numpy as np

# Create a 5x5 array
arr = np.zeros((5, 5), dtype=int)

# Replace border values to 1
arr[0,:] = 1
arr[4,:] = 1
arr[:,0] = 1
arr[:,4] = 1

# Display array
print(arr)