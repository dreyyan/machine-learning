# NumPy Notes
## Concepts
`Broadcasting`: lets you combine arrays of different shapes in operations

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
```
