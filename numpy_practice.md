NumPy Practice Problems
=======================

Getting Started
---------------
- Assume `import numpy as np` at the top of each script or notebook.
- Unless stated otherwise, work with `np.random.default_rng(42)` for reproducibility.
- Prefer vectorized solutions; avoid Python loops unless the task explicitly asks for them.

Level 1 · Fundamentals
----------------------
1. **Array Bootcamp:** Create a 1D array of integers from 10 to 50 (inclusive). Reverse it, compute its mean, and cast the array to `float32`.
2. **Grid Maker:** Build a 5×5 array whose rows are `[0, 1, 2, 3, 4]`. Replace the border values with `1` and the interior with `0`.
3. **Masking 101:** Given `arr = np.arange(20)`, set all odd numbers to `-1` without affecting even numbers. Return a boolean mask of the positions that were changed.
4. **Reshape & Slice:** Create an array of shape `(6, 4)` with values `0..23`. Extract the submatrix containing rows 2–4 and columns 1–3 (0-indexed). Flatten it in column-major order.
5. **Broadcast Warm-up:** For vectors `a = np.arange(5)` and `b = np.arange(5, 10)`, generate a matrix where `M[i, j] = a[i] + b[j]` using broadcasting.

Level 2 · Intermediate Manipulation
-----------------------------------
6. **Fancy Indexing:** With `scores = np.array([83, 91, 77, 65, 88, 92])`, reorder the array by ranking (highest first) and obtain the original indices of the top three scores.
7. **Conditional Substitution:** Given a 4×4 array of random integers in `[0, 9]`, replace all values `< 3` with their square and values `>= 3` with their cube, in-place.
8. **Row Normalization:** Given a matrix `X` of shape `(8, 3)` sampled from a standard normal distribution, normalize each row to unit L2 norm without using loops.
9. **Block Matrix:** Create the 8×8 block matrix `[[A, B], [B, A]]` where `A = np.ones((4, 4))` and `B = np.eye(4) * 3`. Verify the resulting matrix has the expected block structure.
10. **Histogram via Binning:** Simulate 10,000 samples from a normal distribution with mean 2 and std 5. Compute a histogram with bin edges `[-10, -5, 0, 5, 10, 15]` using `np.histogram`. Report counts and relative frequencies.

Level 3 · Advanced Operations
-----------------------------
11. **Rolling Window Mean:** Implement a function `rolling_mean(x, window)` that computes the moving average of a 1D array using vectorized operations (`np.convolve` or stride tricks). Ensure the output aligns with a “valid” window.
12. **Linear Algebra Pipeline:** Generate a random 5×5 matrix `M` (normal distribution) and vector `b` (length 5). Solve `Mx = b`, compute the eigenvalues of `M`, and confirm `M @ M⁻¹` is close to the identity.
13. **Principal Components:** For a matrix `X` with shape `(200, 5)` drawn from a normal distribution, center the columns, compute the covariance matrix, extract the top two eigenvectors, and project the data onto those components.
14. **Broadcasted Distance Matrix:** Given `points` of shape `(N, 3)` and `centers` of shape `(K, 3)`, compute the squared Euclidean distance matrix `D` of shape `(N, K)` without explicit Python loops. Test with `N=500`, `K=4`.
15. **Monte Carlo Integration:** Estimate the integral of `f(x, y) = exp(-(x² + y²))` over the square `[0, 1]²` using 1,000,000 uniform random samples. Compare against the analytical value obtained via numerical integration of the separable form.

Stretch Challenges
-------------------
16. **Image Filters:** Load the 8×8 digits dataset from `sklearn.datasets.load_digits`. Using pure NumPy, implement a simple blur filter (3×3 averaging kernel) and an edge detector (Sobel). Apply both to one sample and display the results.
17. **Custom Broadcasting Op:** Implement a function `pairwise_cosine_similarity(A, B)` where `A` has shape `(m, d)` and `B` has shape `(n, d)`. Return the `(m, n)` cosine similarity matrix using broadcasting and `np.einsum`.
18. **Sparse Interaction:** Given two arrays `a` and `b`, both length 1,000 with 95% zeros, compute all pairwise outer products but only keep entries where both elements are non-zero—in a memory-efficient manner leveraging boolean masks and `np.nonzero`.

Hints & Expected Outcomes
-------------------------
- Level 1 answers should require no loops; check `.dtype`, `.shape`, and `.flatten()` results to validate.
- Level 2 tasks emphasize `np.argsort`, boolean masks, `np.where`, `np.linalg.norm`, and concatenation helpers like `np.block` or `np.vstack`/`np.hstack`.
- Level 3 solutions combine `np.linalg` utilities, broadcasting rules, and efficient reductions; verify with `np.allclose` for numerical checks.
- Stretch problems encourage integration with scikit-learn datasets, `np.pad`, convolution via `np.tensordot`, and mask-based sparsity tricks.
- Consider writing unit tests with `pytest` for each function you implement to ensure repeatability.

Progress Checklist
------------------
- [ ] I can create, reshape, and slice arrays confidently.
- [ ] I can express conditionals, indexing, and broadcasting without loops.
- [ ] I understand aggregation, statistics, and linear algebra routines in `np.linalg`.
- [ ] I can vectorize custom operations and reason about array shapes quickly.
- [ ] I can integrate NumPy with other libraries (scikit-learn, plotting) when needed.

Suggested Workflow
------------------
1. Tackle problems in order. Time-box each level (e.g., 60–90 minutes).
2. After solving, compare your approach to alternative vectorized solutions.
3. Document observations (performance, readability, pitfalls) in a lab notebook.
4. Revisit any problem you solved with loops and attempt a vectorized rewrite.
5. Move to the next level only when the checklist items feel natural.

