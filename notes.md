## **NumPy Fundamentals – Notes & Key Concepts**

---

### **1. Array Bootcamp**
- **Array Creation**: `np.array([...])` → base structure for numerical data
- **Reversing**: `arr[::-1]` → step of `-1` flips order
- **Mean**: average value → `np.mean(arr)`
- **Type Casting**: `np.float32(arr)` or `arr.astype('float32')` → control memory & precision

---

### **2. Grid Maker**
- **Zero Initialization**: `np.zeros(shape, dtype)` → common starting point
- **Border Assignment**: use full-row/column slicing:
  ```python
  arr[0,:] = 1; arr[-1,:] = 1; arr[:,0] = 1; arr[:,-1] = 1
  ```

---

### **3. Masking 101**
- **Boolean Masking**: create logical index → `arr % 2 != 0`
- **Conditional Assignment**: `arr[mask] = value` → modify only selected elements
- **Concept**: **Vectorized filtering** — no loops needed

---

### **4. Reshape & Slice**
- **Reshape**: `arr.reshape(rows, cols)` → change view, not data (must match size)
- **Slicing**: `arr[start:end, start:end]` → extract subregions
- **Flatten (Column-major)**: `.flatten(order='F')` → Fortran-order (column-first)

---

### **5. Broadcast Warm-up**
- **Broadcasting**: automatic alignment of arrays with compatible shapes
  - Rule: dimensions compared **from the right**
  - Example: `(5,1) + (5,)` → `(5,5)`
- **Outer Addition**: `a[:, None] + b` → creates all pairs `a[i] + b[j]`

---

## **Core Concepts & Definitions**

| Concept | Definition | Code Example |
|-------|------------|-------------|
| **Normalization** | Scale a vector so its **L2 norm = 1** (unit length) | `v / np.linalg.norm(v)` |
| **Centering** | Shift data so **mean = 0** per feature | `X - X.mean(axis=0)` |
| **Standardization** | Center + scale to **std = 1** | `(X - mean) / std` |
| **L2 Norm (Euclidean)** | Length of vector: √(Σx²) | `np.linalg.norm(v)` |
| **Softmax** | Convert logits to probabilities (sum to 1) | `exp(x) / sum(exp(x))` |
| **Covariance Matrix** | Measures how features vary together | `(X_centered.T @ X_centered) / (n-1)` |

---

## **Array Creation & Inspection**

| Task | Code | Notes |
|------|------|-------|
| Random [0,1) | `np.random.rand(n)` | Uniform |
| Zeros | `np.zeros((r,c), dtype)` | Initialize weights/masks |
| Linspace | `np.linspace(a,b,n)` | Evenly spaced points |
| Inspect | `.shape`, `.size`, `.dtype` | Always check! |

---

## **Indexing, Slicing & Masking**

- **Submatrix**: `arr[r1:r2, c1:c2]`
- **Replace values**: `arr[arr < 0] = 0`
- **Even numbers**: `arr[arr % 2 == 0]`
- **Main Diagonal**: `np.diagonal(arr)` or `arr.diagonal()`
- **Row filter**: `arr[arr[:, -1] > 0.5]` → last column > 0.5

---

## **Reshaping & Broadcasting**

- **Flatten**: `.flatten()` or `.ravel()` → 1D vector
- **Image → Vector**: `(28,28) → (784,)`
- **Broadcasted Outer Sum**: `a[:, None] + b` → matrix of all pairs
- **Column-wise Ops**:
  ```python
  col_max = X.max(axis=0)
  X_norm = X / col_max
  ```

---

## **Vectorized Computations**

| Task | Formula | Code |
|------|--------|------|
| **MSE** | (1/n)Σ(y - ŷ)² | `np.mean((y_true - y_pred)**2)` |
| **Variance** | E[(x-μ)²] | `np.var(arr, axis=0)` |
| **Center Data** | X ← X - μ | `X -= X.mean(axis=0)` |
| **Softmax** | σ(xᵢ) = e^{xᵢ} / Σe^{xⱼ} | `np.exp(x)/np.exp(x).sum()` |

---

## **Linear Algebra**

| Operation | Code | Notes |
|---------|------|-------|
| **Dot Product** | `v1 @ v2` | Scalar result |
| **Matrix Multiply** | `A @ B` | Shape: (m,p) @ (p,n) → (m,n) |
| **Transpose** | `arr.T` | Flip rows ↔ columns |
| **Inverse** | `np.linalg.inv(A)` | Only if square & full rank |
| **Solve Ax=b** | `np.linalg.solve(A,b)` | Faster than inverse |
| **Projection** | projᵤ(v) = (u·v)/(u·u) × u | ` (np.dot(u,v)/np.dot(u,u)) * u ` |

---

## **Randomness, Sampling & Initialization**

| Task | Code | Purpose |
|------|------|---------|
| **Normal Init** | `np.random.normal(0,1,size)` | Weight initialization |
| **Shuffle Rows** | `np.random.shuffle(arr)` | Randomize data order |
| **Train/Test Split** | `split = int(0.8*len(X))` | Manual split |
| **Reproducibility** | `np.random.seed(42)` | Same results every run |
| **Random Ints** | `np.random.randint(low, high, size)` | Discrete sampling |

---

## **NumPy Best Practices**

- **Vectorize everything** → avoid `for` loops over elements
- **Use `axis=` correctly**:
  - `axis=0` → **down columns** (per feature)
  - `axis=1` → **across rows** (per sample)
- **Prefer `@` over `np.dot` for 2D matrices**
- **Broadcasting > loops**
- **Boolean masks > conditionals**
- **Check shapes early and often**

---

**Final Tip**:  
> **"Think in arrays, not loops."**  
> Every ML operation can be expressed with **reshape + broadcast + mask + reduce**.