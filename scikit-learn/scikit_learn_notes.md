# Scikit-Learn Notes

## Concepts
| Metric        | Question It Answers                                               | Interpretation                                           |
| ------------- | ----------------------------------------------------------------- | -------------------------------------------------------- |
| **Accuracy**  | *Overall, how often was the model correct?*                       | Good for balanced datasets (like Iris).                  |
| **Precision** | *When the model predicts a class, how often is it correct?*       | High precision = few **false positives**.                |
| **Recall**    | *How many actual class examples did the model successfully find?* | High recall = few **false negatives**.                   |
| **F1 Score**  | *Balanced combination of precision and recall.*                   | Best summary measure when working with multiple classes. |
Underfitting
Overfittting

EDA is the part of machine learning where you investigate and understand your dataset before modeling.
The goal is to gather insights that help you decide:

How to preprocess the data (scaling, encoding, handling missing values)

Which features might be useful

Whether the data is balanced

Whether relationships are linear or non-linear

How to handle outliers

| Solver        | Notes                                                                  |
| ------------- | ---------------------------------------------------------------------- |
| `'lbfgs'`     | Default; efficient for multinomial problems, handles L2 regularization |
| `'liblinear'` | Good for small datasets; supports L1 regularization                    |
| `'saga'`      | Supports large datasets, L1/L2/elasticnet, stochastic                  |
| `'newton-cg'` | Efficient for L2 regularization, multinomial                           |


| Dataset size         | Recommended k |
| -------------------- | ------------- |
| Small (<1,000 rows)  | 5–10          |
| Large (>10,000 rows) | 5             |

## Cheatsheet

| Method                | What it does                                                                           | When to use                                        |
| --------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------- |
| **`fit()`**           | Learns (computes) the parameters from data only.                                       | When you only need to *learn* but not *apply* yet. |
| **`fit_transform()`** | Learns the parameters **and immediately applies** the transformation to the same data. | During data preprocessing (training phase).        |
fit() => test data
fit_transform() => train data

| Transformer        | What it does                                               | Use case                                            |
| ------------------ | ---------------------------------------------------------- | --------------------------------------------------- |
| **StandardScaler** | Scales features to **mean=0, std=1**                       | Gradient-based models, SVM, LogisticRegression      |
| **MinMaxScaler**   | Scales features to a **fixed range**, usually 0–1          | Neural networks, when you need bounded input        |
| **MaxAbsScaler**   | Scales features to [-1, 1] based on **max absolute value** | Sparse data                                         |
| **RobustScaler**   | Scales features using **median & IQR**                     | Data with outliers, more robust than StandardScaler |
| **Normalizer**     | Scales **each sample** to unit norm (length=1)             | Text data (TF-IDF), row-wise normalization          |

| Transformer        | What it does                                        | Use case                           |
| ------------------ | --------------------------------------------------- | ---------------------------------- |
| **OneHotEncoder**  | Converts categorical values into **binary columns** | Categorical features for any model |
| **OrdinalEncoder** | Converts categories into **integer codes**          | Ordinal categories with order      |
| **LabelEncoder**   | Converts labels (target) into integers              | Classification targets (0,1,2…)    |
| Transformer            | What it does                                    | Use case                             |
| ---------------------- | ----------------------------------------------- | ------------------------------------ |
| **PolynomialFeatures** | Generates **polynomial & interaction features** | Linear regression, feature expansion |
| **Binarizer**          | Converts values to 0/1 based on a threshold     | Thresholding features                |

| Transformer       | What it does                                                 | Use case                       |
| ----------------- | ------------------------------------------------------------ | ------------------------------ |
| **SimpleImputer** | Fill missing values with **mean, median, mode, or constant** | Handle missing data            |
| **KNNImputer**    | Fill missing values using **nearest neighbors**              | Better for correlated features |

## `sklearn: datasets`

```python
X, y = datasets.{dataset_name}(return_X_y=True)
```

## `sklearn: metrics`
1. 
| `average=` value | Meaning                       | How it Works                                      | When to Use                                  |
| ---------------- | ----------------------------- | ------------------------------------------------- | -------------------------------------------- |
| **`macro`**      | Treat all classes equally     | Average metrics **per class**, no weighting       | ✅ Use when class sizes are *balanced* (Iris) |
| **`micro`**      | Treat all predictions equally | Count total TP/FP/FN and compute metrics globally | Use when class sizes are *imbalanced*        |
| **`weighted`**   | Weight by class frequency     | Like macro, but larger classes contribute more    | Use when classes are *very imbalanced*       |
| **`None`**       | No averaging                  | Returns a metric value **per class**              | Use when you want class-by-class analysis    |

## Additional Notes

1. Standard Deviation
| Situation                                         | Interpretation                  | What to do                              |
| ------------------------------------------------- | ------------------------------- | --------------------------------------- |
| **std ≈ 0**                                       | Feature is **constant**         | Remove feature                          |
| **std is very small compared to others**          | Feature contributes very little | Consider removing or checking relevance |
| **std is much larger than others**                | Feature dominates scaling       | Apply scaling/normalization             |
| **std is reasonable and similar across features** | Feature is balanced             | No preprocessing needed                 |
