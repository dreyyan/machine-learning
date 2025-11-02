{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "cb979c2b-896a-49f9-9c14-de4ed30f3f2e",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "14f45400-4421-4e26-b38d-bb0800cfcbd9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Array: [50. 40. 30. 20. 10.]\n",
      "Mean: 30.0\n"
     ]
    }
   ],
   "source": [
    "# 1. Array Bootcamp\n",
    "# Create 1D array of integers\n",
    "arr = np.array([10, 20, 30, 40, 50])\n",
    "\n",
    "# Reverse the array\n",
    "arr = arr[::-1]\n",
    "\n",
    "# Compute the mean\n",
    "mean = np.mean(arr)\n",
    "\n",
    "# Cast array to 'float32'\n",
    "arr = np.float32(arr)\n",
    "\n",
    "# Display results\n",
    "print(f\"Array: {arr}\")\n",
    "print(f\"Mean: {mean}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "eab04dff-d3c2-4969-a20a-cd3a416929ad",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1 1 1 1 1]\n",
      " [1 0 0 0 1]\n",
      " [1 0 0 0 1]\n",
      " [1 0 0 0 1]\n",
      " [1 1 1 1 1]]\n"
     ]
    }
   ],
   "source": [
    "# 2. Grid Maker\n",
    "# Create a 5x5 array\n",
    "arr = np.zeros((5, 5), dtype=int)\n",
    "\n",
    "# Replace border values to 1\n",
    "arr[0,:] = 1\n",
    "arr[4,:] = 1\n",
    "arr[:,0] = 1\n",
    "arr[:,4] = 1\n",
    "\n",
    "# Display array\n",
    "print(arr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "45dd896a-49ca-4759-bc5c-c7ae29e5fa97",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Arr: [ 0 -1  2 -1  4 -1  6 -1  8 -1 10 -1 12 -1 14 -1 16 -1 18 -1]\n",
      "Mask: [False  True False  True False  True False  True False  True False  True\n",
      " False  True False  True False  True False  True]\n"
     ]
    }
   ],
   "source": [
    "# 3. Masking 101\n",
    "# Create 1D array from 0-19\n",
    "arr = np.arange(20)\n",
    "\n",
    "# Create boolean mask for odd numbers\n",
    "odd_numbers = arr % 2 != 0\n",
    "\n",
    "# Set all odd numbers to '-1'\n",
    "arr[odd_numbers] = -1\n",
    "\n",
    "# Return a boolean mask of the changed positions\n",
    "print(f\"Arr: {arr}\")\n",
    "print(f\"Mask: {odd_numbers}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "abeed938-0391-41c5-bf04-5dbd1bc47d52",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Array: [[ 0  1  2  3]\n",
      " [ 4  5  6  7]\n",
      " [ 8  9 10 11]\n",
      " [12 13 14 15]\n",
      " [16 17 18 19]\n",
      " [20 21 22 23]]\n",
      "Submatrix: [ 9 13 17 10 14 18 11 15 19]\n"
     ]
    }
   ],
   "source": [
    "# 4. Reshape & Slice\n",
    "# Create array of shape (6, 4)\n",
    "arr = np.arange(24).reshape(6, 4)\n",
    "\n",
    "# Extract the submatrix\n",
    "submatrix = arr[2:5, 1:4]\n",
    "\n",
    "# Flatten in column-major order\n",
    "submatrix = submatrix.flatten(order='F')\n",
    "\n",
    "print(f\"Array: {arr}\")\n",
    "print(f\"Submatrix: {submatrix}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "6721a46c-f25a-44c2-8c3d-2e118b5b0e2a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0 1 2 3 4] [5 6 7 8 9]\n",
      "[[ 5  6  7  8  9]\n",
      " [ 6  7  8  9 10]\n",
      " [ 7  8  9 10 11]\n",
      " [ 8  9 10 11 12]\n",
      " [ 9 10 11 12 13]]\n"
     ]
    }
   ],
   "source": [
    "# 5. Broadcast Warm-up\n",
    "a = np.arange(5)\n",
    "b = np.arange(5, 10)\n",
    "\n",
    "# Generate a matrix where 'M[i, j] = a[i] + b[j]' using broadcasting\n",
    "M = a[:, np.newaxis] + b\n",
    "\n",
    "print(a, b)\n",
    "print(M)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "6b4810d4-5b94-4499-a121-705a30fdf79c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.54346102 0.38185747 0.42568327 0.58463394 0.55730101 0.12668906\n",
      " 0.04971912 0.02243986 0.10526145 0.12720365]\n",
      "[[0. 0. 0. 0. 0.]\n",
      " [0. 0. 0. 0. 0.]\n",
      " [0. 0. 0. 0. 0.]\n",
      " [0. 0. 0. 0. 0.]\n",
      " [0. 0. 0. 0. 0.]]\n",
      "float32\n",
      "[0.         0.05263158 0.10526316 0.15789474 0.21052632 0.26315789\n",
      " 0.31578947 0.36842105 0.42105263 0.47368421 0.52631579 0.57894737\n",
      " 0.63157895 0.68421053 0.73684211 0.78947368 0.84210526 0.89473684\n",
      " 0.94736842 1.        ]\n",
      "Shape: (20,)\n",
      "Size: 20\n",
      "Data Type: float64\n"
     ]
    }
   ],
   "source": [
    "# 1. Array Creation & Inspection\n",
    "# Create a 1D array of 10 random floats between 0 and 1\n",
    "arr = np.random.rand(10)\n",
    "print(arr)\n",
    "\n",
    "# Create a (100, 10) array of random floats (simulating 100 samples, 10 features)\n",
    "arr = np.random.rand(100, 10)\n",
    "# print(arr)\n",
    "\n",
    "# Create an array of zeros with shape (5, 5) and dtype float32\n",
    "arr = np.zeros((5, 5), dtype='float32')\n",
    "print(arr)\n",
    "print(arr.dtype)\n",
    "\n",
    "# Generate a sequence from 0 to 1 with 20 equally spaced values.\n",
    "arr = np.linspace(0, 1, 20)\n",
    "print(arr)\n",
    "\n",
    "# Check the shape, size, and dtype of a given array.\n",
    "print(f\"Shape: {arr.shape}\")\n",
    "print(f\"Size: {arr.size}\")\n",
    "print(f\"Data Type: {arr.dtype}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "a70fae4b-e1a7-4160-aa4e-3ea744e27636",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0.10603902 0.86874433 0.38985621 0.56487785]\n",
      " [0.14057616 0.46917026 0.52634341 0.80208548]\n",
      " [0.83979052 0.5183339  0.86482361 0.64757046]\n",
      " [0.22396727 0.02651582 0.81357728 0.34873504]\n",
      " [0.23283107 0.21108728 0.81796863 0.07602402]\n",
      " [0.36478207 0.71505636 0.79078156 0.75654517]]\n",
      "[[0.5183339  0.86482361 0.64757046]\n",
      " [0.02651582 0.81357728 0.34873504]\n",
      " [0.21108728 0.81796863 0.07602402]]\n",
      "[[11  4  8 15]\n",
      " [14  2  6  6]\n",
      " [12  2  5 13]\n",
      " [ 0  0  3 10]]\n",
      "[ 0  2  4  6  8 10 12 14 16 18]\n",
      "[[ 6 14 15 14]\n",
      " [ 8  1  6  7]\n",
      " [12  4  6  2]\n",
      " [14 15  4  7]]\n",
      "Diagonals: [6 1 6 7]\n",
      "[[0.04774367 0.09345202 0.08207665 0.62494583]\n",
      " [0.63870142 0.01213771 0.68563655 0.06712787]\n",
      " [0.31182438 0.48333168 0.11402518 0.58931963]\n",
      " [0.60802707 0.69510123 0.56859409 0.88298823]]\n",
      ">0.5: [[0.04774367 0.09345202 0.08207665 0.62494583]\n",
      " [0.31182438 0.48333168 0.11402518 0.58931963]\n",
      " [0.60802707 0.69510123 0.56859409 0.88298823]]\n"
     ]
    }
   ],
   "source": [
    "# 2. Indexing, Slicing & Masking\n",
    "# Given a (6, 4) matrix, extract rows 2–4 and columns 1–3\n",
    "arr = np.random.rand(24).reshape(6, 4)\n",
    "\n",
    "submatrix = arr[2:5, 1:4]\n",
    "print(arr)\n",
    "print(submatrix)\n",
    "\n",
    "# Replace all negative values in a matrix with 0\n",
    "arr = np.random.randint(16, size=(4, 4))\n",
    "print(arr)\n",
    "\n",
    "# Extract only even numbers from np.arange(20) using masking\n",
    "arr = np.arange(20)\n",
    "\n",
    "even_numbers = arr % 2 == 0\n",
    "\n",
    "print(arr[even_numbers])\n",
    "\n",
    "# From a (5, 5) array, select the diagonal elements\n",
    "arr = np.random.randint(16, size=(4,4))\n",
    "diagonals = np.diagonal(arr)\n",
    "print(arr)\n",
    "print(f\"Diagonals: {diagonals}\")\n",
    "\n",
    "# Select all rows where the last column value is greater than 0.5\n",
    "arr = np.random.rand(16).reshape(4, 4)\n",
    "greater = arr[:, -1] > 0.5\n",
    "\n",
    "print(arr)\n",
    "print(f\">0.5: {arr[greater]}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 121,
   "id": "cc0d3028-a09d-4cd9-9f09-359ddbe3aafa",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(784,)\n",
      "[10 19 14 11  1 22  6 12  1 21  0  2  2  0  7 24 23 22 24 21 23 26 20  3\n",
      " 19 21 18]\n",
      "[0 1 2]\n",
      "[3 4 5]\n",
      "[[3 4 5]\n",
      " [4 5 6]\n",
      " [5 6 7]]\n",
      "[[1. 5. 4.]\n",
      " [3. 6. 6.]\n",
      " [3. 6. 8.]]\n",
      "3.0\n",
      "6.0\n",
      "8.0\n",
      "Before standardization: [[0.77114727 0.39242964 0.7113464 ]\n",
      " [0.17945744 0.01189358 0.5766388 ]\n",
      " [0.22445571 0.71373999 0.30966598]]\n",
      "After standardization: [[ 1.41091064  0.06881892  1.0711325 ]\n",
      " [-0.78911165 -1.25770336  0.26412518]\n",
      " [-0.62179899  1.18888444 -1.33525768]]\n"
     ]
    }
   ],
   "source": [
    "# 3. Reshaping & Broadcasting\n",
    "# Reshape an array of shape (28, 28) into (784,)\n",
    "arr = np.random.randint(784, size=(28,28)).flatten()\n",
    "print(arr.shape)\n",
    "\n",
    "# Flatten a (3, 3, 3) array into a 1D vector\n",
    "arr = np.random.randint(27, size=(3, 3, 3))\n",
    "arr = arr.flatten()\n",
    "print(arr)\n",
    "\n",
    "# Given a = np.arange(3) and b = np.arange(3, 6), create a matrix where M[i, j] = a[i] + b[j] using broadcasting\n",
    "a = np.arange(3)\n",
    "b = np.arange(3, 6)\n",
    "\n",
    "M = a[:, np.newaxis] + b\n",
    "print(a)\n",
    "print(b)\n",
    "print(M)\n",
    "\n",
    "# Normalize each column of a matrix by dividing by its column max\n",
    "arr = np.random.randint(9, size=(3,3)).astype('float64')\n",
    "print(arr)\n",
    "for i in range(arr.shape[1]):\n",
    "    col_max = np.max(arr[:,i])\n",
    "    arr[:,i] /= col_max\n",
    "    print(col_max)\n",
    "\n",
    "# Standardize each column: subtract mean and divide by standard deviation\n",
    "arr = np.random.rand(3, 3)\n",
    "print(f\"Before standardization: {arr}\")\n",
    "for i in range(arr.shape[1]):\n",
    "    mean = np.mean(arr[:,i])\n",
    "    std = np.std(arr[:,i])\n",
    "    arr[:,i] = (arr[:,i] - mean) / std\n",
    "print(f\"After standardization: {arr}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 144,
   "id": "5f6d426f-388c-4d44-9cb6-5377c1f3ddc5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[9 6 1 1]\n",
      " [2 9 8 8]\n",
      " [6 6 1 4]\n",
      " [4 5 7 2]]\n",
      "Mean: [5.25 6.5  4.25 3.75]\n",
      "Standard Deviation: [2.58602011 1.5        3.26917421 2.68095132]\n",
      "Variance: [ 6.6875  2.25   10.6875  7.1875]\n",
      "True: [0 0 0 1 1 1 0 0 1]\n",
      "Predictions: [1 1 0 1 0 0 1 0 1]\n",
      "MSE (Mean-Squared Error): 0.5555555555555556\n",
      "Euclidean Norm: 7.810249675906654]\n",
      "Normalized: [0.6401844  0.76822128]\n",
      "Norm of normalized vector: 1.0\n",
      "Original:\n",
      "[[6. 4. 6.]\n",
      " [5. 6. 3.]\n",
      " [7. 3. 8.]]\n",
      "Centered:\n",
      "[[ 0.         -0.33333333  0.33333333]\n",
      " [-1.          1.66666667 -2.66666667]\n",
      " [ 1.         -1.33333333  2.33333333]]\n",
      "SoftMax: [0.65900114 0.24243297 0.09856589]\n"
     ]
    }
   ],
   "source": [
    "# 4. Vectorized Computations\n",
    "# Compute the mean, standard deviation, and variance of a dataset along each feature (axis=0)\n",
    "arr = np.random.randint(10, size=(4,4))\n",
    "mean: float = np.mean(arr, axis=0)\n",
    "std: float = np.std(arr, axis=0)\n",
    "variance: float = np.var(arr, axis=0)\n",
    "\n",
    "print(arr)\n",
    "print(f\"Mean: {mean}\\nStandard Deviation: {std}\\nVariance: {variance}\")\n",
    "\n",
    "# Given two arrays y_true and y_pred, compute mean squared error (MSE) without loops\n",
    "y_true = np.random.randint(2, size=(9,))\n",
    "y_pred = np.random.randint(2, size=(9,))\n",
    "mse = np.mean((y_true - y_pred) ** 2)\n",
    "print(f\"True: {y_true}\\nPredictions: {y_pred}\\nMSE (Mean-Squared Error): {mse}\")\n",
    "\n",
    "# Normalize a vector v so that its Euclidean norm = 1\n",
    "v = [5, 6]\n",
    "norm = np.linalg.norm(v)\n",
    "print(f\"Euclidean Norm: {norm}]\")\n",
    "v_normalized = v / norm\n",
    "\n",
    "print(f\"Normalized: {v_normalized}\")\n",
    "print(\"Norm of normalized vector:\", np.linalg.norm(v_normalized))\n",
    "\n",
    "# Center a dataset by subtracting its mean along each column\n",
    "arr = np.random.randint(9, size=(3, 3)).astype('float64')\n",
    "print(f\"Original:\\n{arr}\")\n",
    "for i in range(arr.shape[1]):\n",
    "    mean = np.mean(arr[:,i])\n",
    "    arr[:,i] -= mean\n",
    "print(f\"Centered:\\n{arr}\")\n",
    "\n",
    "# Compute softmax for a 1D array\n",
    "arr = np.array([2.0, 1.0, .1])\n",
    "\n",
    "exp_arr = np.exp(arr)\n",
    "softmax_arr = exp_arr / np.sum(exp_arr)\n",
    "print(f\"SoftMax: {softmax_arr}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 157,
   "id": "a265e4bc-f949-474c-9c02-15aadad3d36f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Dot Product: 190\n",
      "Vector #1:\n",
      "[[0 4]\n",
      " [2 2]]\n",
      "Vector #2:\n",
      "[[4 4]\n",
      " [2 3]]\n",
      "Matrix Multiplication:\n",
      "[[ 8 12]\n",
      " [12 14]]\n",
      "Transpose: [[1 4]\n",
      " [2 5]\n",
      " [3 6]]\n"
     ]
    }
   ],
   "source": [
    "# 5. Linear Algebra\n",
    "# Compute the dot product of two vectors\n",
    "v1 = np.array([2, 4, 6, 8, 10])\n",
    "v2 = np.array([1, 3, 5, 7, 9])\n",
    "\n",
    "dot_product = np.dot(v1, v2)\n",
    "print(f\"Dot Product: {dot_product}\")\n",
    "\n",
    "# Multiply two matrices A and B (where shapes align)\n",
    "v1 = np.random.randint(5, size=(2,2))\n",
    "v2 = np.random.randint(5, size=(2,2))\n",
    "\n",
    "matrix_mult = v1 @ v2\n",
    "print(f\"Vector #1:\\n{v1}\")\n",
    "print(f\"Vector #2:\\n{v2}\")\n",
    "print(f\"Matrix Multiplication:\\n{matrix_mult}\")\n",
    "\n",
    "# Compute the transpose of a matrix\n",
    "arr = np.array([\n",
    "    [1, 2, 3],\n",
    "    [4, 5, 6]\n",
    "])\n",
    "arr_transpose = arr.T\n",
    "print(f\"Transpose: {arr_transpose}\")\n",
    "\n",
    "# Find the inverse of a 3×3 matrix\n",
    "\n",
    "\n",
    "# Compute the Euclidean norm (L2 norm) of a vector\n",
    "\n",
    "\n",
    "# Compute the covariance matrix of a dataset X using X.T @ X / n\n",
    "\n",
    "\n",
    "# Project a vector v onto another vector u\n",
    "\n",
    "\n",
    "# Solve a linear system Ax = b using np.linalg.solve\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
