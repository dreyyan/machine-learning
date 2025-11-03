NumPy Cheatsheet
=================

Setup & Basics
--------------
- `import numpy as np`
- Array creation: `np.array(list_like)`, `np.asarray(obj)`, `np.zeros(shape)`, `np.ones(shape)`, `np.empty(shape)`, `np.full(shape, fill_value)`
- Ranges: `np.arange(start, stop, step)`, `np.linspace(start, stop, num, endpoint=True)`, `np.logspace(start, stop, num, base=10.0)`
- Identity & diag: `np.eye(N, M=None, k=0)`, `np.identity(n)`, `np.diag(v, k=0)`, `np.diagflat(v, k=0)`
- Random init: `np.random.default_rng(seed)`, `rng.random(shape)`, `rng.normal(loc, scale, size)`, `rng.integers(low, high=None, size=None)`
- Dtypes: `array.dtype`, `array.astype(new_dtype)`, `np.iinfo(dtype)`, `np.finfo(dtype)`
- Inspect: `array.shape`, `array.ndim`, `array.size`, `array.itemsize`, `array.nbytes`, `array.dtype`

Indexing & Slicing
------------------
- Basic slicing: `array[start:stop:step]`, `array[..., idx]`, `array[:, i]`
- Fancy indexing: `array[[row_indices], [col_indices]]`, `array[np.array(mask)]`
- Boolean masking: `array[array > 0]`, `np.where(condition, x, y)`, `np.argwhere(condition)`
- Index helpers: `np.take(array, indices, axis=None)`, `np.put(array, indices, values)`
- Iteration: `np.nditer(array)`, `np.ndenumerate(array)`

Shape Manipulation
------------------
- Reshape: `array.reshape(new_shape)`, `array.ravel(order='C')`, `array.flatten(order='C')`
- Transpose: `array.T`, `array.transpose(axes)`, `np.swapaxes(array, axis1, axis2)`, `np.moveaxis(array, source, destination)`
- Expand/Reduce dims: `np.expand_dims(array, axis)`, `np.squeeze(array, axis=None)`
- Joining: `np.concatenate(seq, axis=0)`, `np.stack(seq, axis=0)`, `np.column_stack(tup)`, `np.row_stack(tup)`, `np.hstack(tup)`, `np.vstack(tup)`
- Splitting: `np.split(array, indices_or_sections, axis=0)`, `np.array_split(array, sections, axis=0)`

Elementwise Operations
----------------------
- Arithmetic: `+`, `-`, `*`, `/`, `**`, `np.add`, `np.subtract`, `np.multiply`, `np.divide`, `np.mod`, `np.power`
- Comparison: `np.equal`, `np.not_equal`, `np.greater`, `np.greater_equal`, `np.less`, `np.less_equal`
- Logical: `np.logical_and`, `np.logical_or`, `np.logical_not`, `np.logical_xor`
- Bitwise: `np.bitwise_and`, `np.bitwise_or`, `np.bitwise_xor`, `np.invert`
- Unary: `np.negative`, `np.abs`, `np.sign`, `np.sqrt`, `np.square`
- Rounding: `np.round`, `np.floor`, `np.ceil`, `np.trunc`, `np.fix`
- Special: `np.exp`, `np.log`, `np.log10`, `np.log2`, `np.sin`, `np.cos`, `np.tan`, `np.arcsin`, `np.sinh`, `np.cosh`

Aggregations & Statistics
-------------------------
- Reductions: `np.sum(array, axis=None)`, `np.prod`, `np.mean`, `np.median`, `np.std`, `np.var`, `np.min`, `np.max`
- Axis-wise: `array.sum(axis=0)`, `array.mean(axis=1, keepdims=True)`
- Order stats: `np.percentile`, `np.quantile`, `np.nanpercentile`, `np.nanquantile`
- Nan-aware: `np.nanmean`, `np.nanstd`, `np.nanmin`, `np.nanmax`, `np.nansum`
- Counts: `np.count_nonzero`, `np.unique(return_counts=True)`, `np.bincount`
- Accumulations: `np.cumsum`, `np.cumprod`, `np.diff`, `np.ediff1d`

Linear Algebra (`np.linalg`)
---------------------------
- Matrix products: `np.dot(a, b)`, `a @ b`, `np.matmul(a, b)`, `np.vdot(a, b)`
- Decompositions: `np.linalg.eig`, `np.linalg.eigh`, `np.linalg.svd`, `np.linalg.qr`, `np.linalg.cholesky`
- Solvers: `np.linalg.solve(A, b)`, `np.linalg.lstsq(A, b)`, `np.linalg.inv(A)`, `np.linalg.pinv(A)`
- Norms & properties: `np.linalg.norm`, `np.trace`, `np.linalg.det`, `np.linalg.matrix_rank`
- Special ops: `np.outer(a, b)`, `np.inner(a, b)`, `np.cross(a, b)`

Random (`numpy.random` / Generator)
----------------------------------
- Create generator: `rng = np.random.default_rng(seed)`
- Uniform: `rng.random(size)`, `rng.uniform(low, high, size)`
- Normal: `rng.normal(loc, scale, size)`
- Discrete: `rng.integers(low, high, size)`, `rng.choice(a, size=None, replace=True, p=None)`
- Other distributions: `rng.binomial`, `rng.poisson`, `rng.beta`, `rng.gamma`, `rng.multivariate_normal`
- Shuffling: `rng.shuffle(array)`, `rng.permutation(array or n)`

Logical & Set Operations
------------------------
- Truth checks: `np.all(array, axis=None)`, `np.any(array, axis=None)`
- Comparisons: `np.isclose`, `np.allclose`, `np.array_equal`
- Null checks: `np.isnan`, `np.isfinite`, `np.isinf`
- Set logic: `np.intersect1d`, `np.union1d`, `np.setdiff1d`, `np.setxor1d`
- Membership: `np.in1d`, `np.isin`

Sorting & Searching
-------------------
- Sorting: `np.sort(array, axis=-1)`, `np.argsort(array, axis=-1)`, `np.lexsort(keys)`
- Partition: `np.partition(array, kth, axis=-1)`, `np.argpartition`
- Extremes: `np.argmax(array, axis=None)`, `np.argmin`
- Search: `np.searchsorted(sorted_array, values, side='left')`, `np.nonzero(array)`

Broadcasting & Tiling
---------------------
- Broadcasting rules: trailing dimensions of size 1 expand to match
- Helpers: `np.broadcast_to(array, shape)`, `np.broadcast_arrays(*arrays)`
- Repetition: `np.tile(array, reps)`, `np.repeat(array, repeats, axis=None)`

Structured Arrays & Records
---------------------------
- Define dtype: `dt = np.dtype([('field1', np.float32), ('field2', np.int32)])`
- Create: `np.array(list_of_tuples, dtype=dt)`, `np.zeros(shape, dtype=dt)`
- Access fields: `array['field1']`
- Rec functions: `np.rec.array(seq, dtype=None)`

Input / Output
--------------
- Text: `np.loadtxt(filename, delimiter=',')`, `np.genfromtxt(filename, delimiter=',', dtype=float)`
- Binary: `np.save(filename, array)`, `np.savez(filename, array1=a, array2=b)`, `np.load(filename)`
- Memory: `np.memmap(filename, dtype, mode, shape)`
- String formatting: `np.array_str(array)`, `np.array2string(array, precision=3)`

Utilities & Helpers
-------------------
- Copying: `array.copy(order='C')`, `np.copyto(dst, src)`
- Type info: `array.view(new_dtype)`, `np.frombuffer(buffer, dtype, count=-1, offset=0)`
- Creation from iterables: `np.fromiter(iterable, dtype, count=-1)`, `np.fromfunction(function, shape, dtype=int)`
- Mesh/grid: `np.meshgrid(*arrays, indexing='xy')`, `np.mgrid[start:stop:step]`, `np.ogrid` (open grid)
- Padding: `np.pad(array, pad_width, mode='constant', constant_values=0)`
- Clip: `np.clip(array, a_min, a_max, out=None)`, `np.interp(x, xp, fp)`
- Vectorize: `np.vectorize(pyfunc, otypes=None)`, `np.apply_along_axis(func, axis, array)`

Performance Tips
----------------
- Prefer vectorized operations to Python loops
- Use `np.einsum` for complex tensor contractions
- Exploit `out=` parameters to avoid allocations, e.g. `np.add(a, b, out=result)`
- For large datasets consider `np.memmap`, `numexpr`, or switching to `CuPy` for GPU arrays
- Profile with `%timeit` in IPython/Jupyter and the `perf_counter` from `time`

Further Reading
---------------
- Official docs: https://numpy.org/doc/stable/
- NumPy Reference: https://numpy.org/doc/stable/reference/
- Broadcasting tutorial: https://numpy.org/doc/stable/user/basics.broadcasting.html

