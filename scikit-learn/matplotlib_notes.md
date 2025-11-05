```python
# 1. Histogram (feature distribution)
df.hist(figsize=(8,6))
plt.show()

# 2. Scatterplot (compare two features)
plt.scatter(df['sepal length (cm)'], df['petal length (cm)'], c=y)
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.show()

# 3. Pairplot (relationships between all features)
import seaborn as sns
sns.pairplot(df.join(pd.Series(y, name='species')), hue='species')
plt.show()

# 4. Confusion Matrix (model performance visualization)
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.show()

```