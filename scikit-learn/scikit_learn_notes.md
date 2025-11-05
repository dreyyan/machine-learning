# Pandas Notes

## Concepts
| Metric        | Question It Answers                                               | Interpretation                                           |
| ------------- | ----------------------------------------------------------------- | -------------------------------------------------------- |
| **Accuracy**  | *Overall, how often was the model correct?*                       | Good for balanced datasets (like Iris).                  |
| **Precision** | *When the model predicts a class, how often is it correct?*       | High precision = few **false positives**.                |
| **Recall**    | *How many actual class examples did the model successfully find?* | High recall = few **false negatives**.                   |
| **F1 Score**  | *Balanced combination of precision and recall.*                   | Best summary measure when working with multiple classes. |


## Cheatsheet

```python

```

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
