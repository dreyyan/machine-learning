| Option                       | Description                                                                            | When to Use                                                       |
| ---------------------------- | -------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **1. Remove them**           | Drop rows outside the IQR range (e.g. beyond 1.5×IQR).                                 | When you know they’re errors or irrelevant.                       |
| **2. Cap (clip) them**       | Replace values below/above thresholds with Q1–1.5×IQR and Q3+1.5×IQR.                  | When you want to keep the row but limit impact.                   |
| **3. Transform them**        | Use log, sqrt, or Box-Cox transformation to reduce skew.                               | When values are valid but highly skewed.                          |
| **4. Impute them**           | Replace outliers with median or mean.                                                  | When few outliers exist in important features.                    |
| **5. Model-robust handling** | Use algorithms not sensitive to outliers (e.g. Decision Tree, Random Forest, XGBoost). | When you don’t want to manually remove outliers.                  |
| **6. Leave them**            | Keep all data points as-is.                                                            | When outliers are real and informative (e.g., anomaly detection). |
