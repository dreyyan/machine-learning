import numpy as np

# Create 1D array of integers
arr = np.array([10, 20, 30, 40, 50])

# Reverse the array
arr = arr[::-1]

# Compute the mean
mean = np.mean(arr)

# Cast array to 'float32'
arr = np.float32(arr)

# Display results
print(f"Array: {arr}")
print(f"Mean: {mean}")