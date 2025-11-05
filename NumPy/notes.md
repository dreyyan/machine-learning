# NumPy Notes

## Concepts

`Broadcasting`: lets you combine arrays of different shapes in operations
`Standardization`: process of transforming data so that each feature (column) has a mean = 0, std = 1
`Normalization`: rescales data so that all values fall within a specific range — usually 0 to 1
`Rolling Standard Deviation`: measures how much values fluctuate within a moving window of fixed size

## Cheatsheet

```python
""" Attributes """
# PI
np.pi

""" Methods """
np.zeros(shape, dtype=float, order='C')

# create a 2D array with ones on a specified diagonal and zeros elsewhere
np.eye(N, M=None, k=0, dtype=float)

# generate evenly spaced numbers over a specified interval
np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)

# create array filled with a constant value
np.full(shape, fill_value)

# generate an array of random floats
np.random.rand(d0, d1, ..., dn)

# creates an open mesh (broadcastable) from multiple 1D index arrays
np.ix_(row_indices, col_indices, ...)

# flip array along a specific
np.flip(array, axis=axis_number)

# create a diagonal matrix
np.diag(v, k=0)

# generates random numbers following a bell curve (Gaussian distribution)
np.random.normal(loc=0.0, scale=1.0, size=None)

# reorder axes of an array
np.transpose(a, axes=None) # (axes = a tuple/list of integers specifying the new order of all axes)

# stack matrices horizontally/vertically
np.hstack/np.vstack([a,b,c,...])

# repeats array along each dimension
np.tile(arr, reps)
```

## `scipy.stats`

```python
#
# calculates mode
mode(array, axis=0, nan_policy='propagate', keepdims: bool = False)

```

## Additional Notes

1. col-wise => axis=0, row-wise => axis=1
