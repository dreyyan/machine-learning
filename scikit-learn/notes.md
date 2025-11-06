```markdown
# Machine Learning Notes – Consolidated README

## Table of Contents

- [1. Pandas](#1-pandas)
  - [1.1 Concepts](#11-concepts)
  - [1.2 Cheatsheet](#12-cheatsheet)
  - [1.3 `pd.read_csv` Parameters](#13-pdread_csv-parameters)
  - [1.4 Downcasting & Value Counts](#14-downcasting--value-counts)
- [2. NumPy](#2-numpy)
  - [2.1 Concepts](#21-concepts)
  - [2.2 Cheatsheet](#22-cheatsheet)
  - [2.3 `scipy.stats.mode`](#23-scipystatsmode)
  - [2.4 Axis Convention](#24-axis-convention)
- [3. Scikit-Learn](#3-scikit-learn)
  - [3.1 Core Concepts](#31-core-concepts)
    - [Evaluation Metrics](#evaluation-metrics)
    - [Underfitting & Overfitting](#underfitting--overfitting)
    - [EDA (Exploratory Data Analysis)](#eda-exploratory-data-analysis)
  - [3.2 Logistic Regression Solvers](#32-logistic-regression-solvers)
  - [3.3 Cross-Validation: Choosing `k`](#33-cross-validation-choosing-k)
  - [3.4 `fit()` vs `fit_transform()`](#34-fit-vs-fit_transform)
  - [3.5 Transformers](#35-transformers)
    - [Scaling](#scaling-transformers)
    - [Encoding](#encoding-transformers)
    - [Feature Engineering](#feature-engineering-transformers)
    - [Imputation](#imputation-transformers)
  - [3.6 `sklearn.datasets`](#36-sklearndatasets)
  - [3.7 `sklearn.metrics` Averaging Strategies](#37-sklearnmetrics-averaging-strategies)
  - [3.8 Standard Deviation Guidelines](#38-standard-deviation-guidelines)
- [4. Matplotlib & Seaborn](#4-matplotlib--seaborn)
  - [4.1 Visualization Code Snippets](#41-visualization-code-snippets)

---

<a name="1-pandas"></a>
## 1. Pandas

<a name="11-concepts"></a>
### 1.1 Concepts
- **`Downcasting`**: Change column dtype to a smaller, memory-efficient type **without losing information** (e.g., `float64` → `float32`, `int64` → `int8`).

<a name="12-cheatsheet"></a>
### 1.2 Cheatsheet

```python
# Attributes
df.shape          # (rows, columns)
df.columns        # column names

# Methods
df.head(n=5)      # first n rows
df.tail(n=5)      # last n rows
df.info()         # dtypes, non-null counts
df.describe()     # summary stats

# Indexing
df.loc[row_labels, col_labels]           # label-based
df.loc[(cond1) & (cond2)]
df.iloc[row_pos, col_pos]                # position-based
df.query("col > 5 and col2 == 'A'")      # string query

# I/O
df.to_csv("file.csv", index=False)
```

<a name="13-pdread_csv-parameters"></a>
### 1.3 `pd.read_csv` Parameters

```python
pd.read_csv(
    filepath_or_buffer,   # str or file-like
    sep=',',              # delimiter
    header='infer',       # row for column names
    names=None,           # custom column names
    index_col=None,       # column(s) as index
    usecols=None,         # load specific columns
    dtype=None,           # force dtypes
    skiprows=None,        # skip initial rows
    nrows=None,           # read only n rows
    na_values=None,       # extra NA strings
    parse_dates=False,    # auto-parse dates
    infer_datetime_format=False
)
```

<a name="14-downcasting--value-counts"></a>
### 1.4 Downcasting & Value Counts

```python
df.astype('category')                    # categorical → category
pd.to_numeric(df['col'], downcast='float')  # or 'integer', 'signed', 'unsigned'

df['category_col'].value_counts()        # frequency of each category
```

[↑ Back to TOC](#table-of-contents)

---

<a name="2-numpy"></a>
## 2. NumPy

<a name="21-concepts"></a>
### 2.1 Concepts
| Term | Description |
|------|-----------|
| **Broadcasting** | Combine arrays of different shapes in arithmetic |
| **Standardization** | Scale to `mean=0`, `std=1` |
| **Normalization** | Rescale to range (usually `[0,1]`) |
| **Rolling Std** | Fluctuation in a moving window |

<a name="22-cheatsheet"></a>
### 2.2 Cheatsheet

```python
# Constants
np.pi

# Creation
np.ones(shape, dtype=float)
np.zeros(shape, dtype=float)
np.eye(N, M=None, k=0, dtype=float)           # identity / diagonal
np.linspace(start, stop, num=50, endpoint=True)
np.full(shape, fill_value)
np.random.rand(d0, d1, ...)                   # uniform [0,1)
np.random.normal(loc=0.0, scale=1.0, size=None)

# Selection & Reshaping
np.random.choice(a, size=None, replace=True, p=None)
np.transpose(a, axes=None)
np.flip(arr, axis=0)

# Stacking
np.hstack([a,b])  np.vstack([a,b])

# Advanced
np.ix_(rows, cols)       # open mesh for indexing
np.tile(arr, reps)       # repeat array
np.diag(v, k=0)          # diagonal matrix
```

<a name="23-scipystatsmode"></a>
### 2.3 `scipy.stats.mode`

```python
from scipy.stats import mode
mode(array, axis=0, nan_policy='propagate', keepdims=False)
```

<a name="24-axis-convention"></a>
### 2.4 Axis Convention
- `axis=0` → **row-wise** (down columns)
- `axis=1` → **column-wise** (across rows)

[↑ Back to TOC](#table-of-contents)

---

<a name="3-scikit-learn"></a>
## 3. Scikit-Learn

<a name="31-core-concepts"></a>
### 3.1 Core Concepts

#### Evaluation Metrics
| Metric | Question | Interpretation |
|--------|----------|----------------|
| **Accuracy** | *Overall, how often correct?* | Good for **balanced** data |
| **Precision** | *When predicted positive, how often right?* | Low FP |
| **Recall** | *How many actual positives found?* | Low FN |
| **F1 Score** | *Harmonic mean of precision & recall* | Best for **imbalanced/multi-class** |

#### Underfitting & Overfitting
- **Underfitting**: Model too simple → poor train & test performance  
- **Overfitting**: Model too complex → great on train, poor on test

#### EDA (Exploratory Data Analysis)
Goals:
- Preprocessing decisions (scaling, encoding, missing values)
- Feature importance
- Class balance
- Linearity vs non-linearity
- Outlier detection

<a name="32-logistic-regression-solvers"></a>
### 3.2 Logistic Regression Solvers

| Solver | Notes |
|--------|-------|
| `'lbfgs'` | Default; L2; efficient for multinomial |
| `'liblinear'` | Small datasets; supports L1 |
| `'saga'` | Large datasets; L1/L2/elasticnet; stochastic |
| `'newton-cg'` | L2; efficient for multinomial |

<a name="33-cross-validation-choosing-k"></a>
### 3.3 Cross-Validation: Choosing `k`

| Dataset Size | Recommended `k` |
|--------------|-----------------|
| Small (<1k)  | 5–10 |
| Large (>10k) | 5 |

<a name="34-fit-vs-fit_transform"></a>
### 3.4 `fit()` vs `fit_transform()`

| Method | Behavior | When to Use |
|--------|----------|-------------|
| `fit()` | Learns parameters only | On **test data** or when splitting |
| `fit_transform()` | Learns **and applies** | On **training data** during preprocessing |

> **Rule**: `fit_transform()` → **train**; `transform()` → **test**

<a name="35-transformers"></a>
### 3.5 Transformers

#### Scaling Transformers
| Transformer | Transformation | Use Case |
|-----------|----------------|----------|
| **StandardScaler** | Mean=0, Std=1 | Gradient-based (LR, SVM, NN) |
| **MinMaxScaler** | [0,1] or custom | Neural nets, bounded input |
| **MaxAbsScaler** | [-1,1] via max abs | Sparse data |
| **RobustScaler** | Median & IQR | Outliers present |
| **Normalizer** | Unit norm per **row** | Text (TF-IDF), row-wise |

#### Encoding Transformers
| Transformer | Output | Use Case |
|-----------|--------|----------|
| **OneHotEncoder** | Binary columns | Any categorical |
| **OrdinalEncoder** | Integer codes | Ordered categories |
| **LabelEncoder** | Integer labels | **Target only** |

#### Feature Engineering Transformers
| Transformer | Output | Use Case |
|-----------|--------|----------|
| **PolynomialFeatures** | Polynomial + interaction terms | Linear models needing curvature |
| **Binarizer** | 0/1 via threshold | Threshold features |

#### Imputation Transformers
| Transformer | Method | Use Case |
|-----------|--------|----------|
| **SimpleImputer** | mean/median/mode/constant | Basic missing value fill |
| **KNNImputer** | k-nearest neighbors | Correlated features |

<a name="36-sklearndatasets"></a>
### 3.6 `sklearn.datasets`

```python
from sklearn import datasets
X, y = datasets.load_iris(return_X_y=True)  # or any dataset
```

<a name="37-sklearnmetrics-averaging-strategies"></a>
### 3.7 `sklearn.metrics` Averaging Strategies

| `average=` | Meaning | How It Works | When to Use |
|------------|---------|--------------|-------------|
| **`macro`** | Equal weight per class | Avg per-class metrics | **Balanced** classes |
| **`micro`** | Equal weight per sample | Global TP/FP/FN | **Imbalanced** |
| **`weighted`** | Weight by support | Like macro, but scaled | **Very imbalanced** |
| **`None`** | Per-class array | No averaging | Class-level analysis |

<a name="38-standard-deviation-guidelines"></a>
### 3.8 Standard Deviation Guidelines

| Std Situation | Interpretation | Action |
|---------------|----------------|--------|
| `std ≈ 0` | Constant feature | **Remove** |
| `std << others` | Low variance | Consider removal |
| `std >> others` | Dominates scale | **Scale** |
| `std` similar across | Balanced | No action |

[↑ Back to TOC](#table-of-contents)

---

<a name="4-matplotlib--seaborn"></a>
## 4. Matplotlib & Seaborn

<a name="41-visualization-code-snippets"></a>
### 4.1 Visualization Code Snippets

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# 1. Histogram (feature distribution)
df.hist(figsize=(8,6))
plt.show()

# 2. Scatterplot (two features)
plt.scatter(df['sepal length (cm)'], df['petal length (cm)'], c=y)
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.show()

# 3. Pairplot (all features + hue)
sns.pairplot(df.join(pd.Series(y, name='species')), hue='species')
plt.show()

# 4. Confusion Matrix
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.show()
```

[↑ Back to TOC](#table-of-contents)