import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from pandas import DataFrame
from typing import Tuple

# [FUNCTION] Return DataFrame, features, and targets of dataset
def load_dataset(dataset) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Load a scikit-learn dataset into pandas DataFrames.

    Parameters
    ----------
    dataset : sklearn.utils.Bunch
        A scikit-learn dataset object containing `.data`, `.feature_names`, and `.target` attributes.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, pd.Series)
        A tuple containing:
        - **df** : pandas.DataFrame  
          The complete dataset, including all features and the target column.  
        - **X** : pandas.DataFrame  
          The features (independent variables) of the dataset.  
        - **y** : pandas.Series  
          The target (dependent variable) of the dataset.
    """
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['target'] = dataset.target
    X = df.drop(columns=['target'], axis=1)
    y = df['target']
    return df, X, y

# [FUNCTION] Display dataset information (e.g. statistics)
def display_dataset_info(df, features=False, target=False, null_count=False, class_dist=False, info=False, describe=False) -> None:
    """
    Display dataset information for exploratory data analysis (EDA).

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset containing features and a 'target' column.
    features : bool, optional
        If True, display the dataset's feature names.
    target : bool, optional
        If True, display the target column name and its unique values.
    null_count : bool, optional
        If True, display the number of missing values per column.
    class_dist : bool, optional
        If True, display the class distribution of the target variable.
    info : bool, optional
        If True, display general dataset information (e.g., column types, non-null counts).
    describe : bool, optional
        If True, display summary statistics (mean, std, quartiles, etc.).

    Returns
    -------
    None
        This function prints information to the console and does not return any value.
    """
    print(f"{' [ DATASET INFORMATION ] ':=^40}")
    
    if features:
        print(f"\n{' [ Features ] ':-^40}")
        for name in df.drop(columns=['target']).columns.to_list():
            print(f"• {name}")
            
    if target:
        print(f"\n{' [ Target ] ':-^40}")
        print(f"• {df['target'].name}")

    if null_count:
        print(f"\n{' [ Null Count ] ':-^40}")
        null_counts = df.isnull().sum()
        print(null_counts)

    if class_dist:
        class_counts = df['target'].value_counts()
        class_percent = df['target'].value_counts(normalize=True) * 100
        print(f"\n{' [ Class Distribution ] ':-^40}")
        for cls in class_counts.index:
            print(f"• {cls}: {class_counts[cls]} samples ({class_percent[cls]:.2f}%)")
        

    if info:
        print(f"\n{' [ Info ] ':-^40}")
        print(f"{df.info()}\n")
        
    if describe:
        print(f"\n{' [ Description ] ':-^40}")
        print(f"{df.describe()}\n")

# [FUNCTION] Display metrics for model performance
def evaluate_model(model, X_test, y_test):
    """
    [Description]
    Evaluates the performance of a trained machine learning model on test data 
    using standard classification metrics.

    [Params]
    model: trained scikit-learn model with a .predict() method
    X_test: test feature set (DataFrame or array-like)
    y_test: true labels for the test set (Series or array-like)

    [Returns]
    dict: a dictionary containing performance metrics:
        - "Accuracy": overall correctness of the model
        - "Precision": average precision across classes (macro)
        - "Recall": average recall across classes (macro)
        - "F1-Score": harmonic mean of precision and recall (macro)
    """
    y_pred = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro"),
        "Recall": recall_score(y_test, y_pred, average="macro"),
        "F1-Score": f1_score(y_test, y_pred, average="macro")
    }

# [FUNCTION] Plot metrics for model performance
def plot_metrics(metrics_dict) -> None:
    """
    [Description]
    Plots a bar chart of model performance metrics for easy visual comparison.

    [Params]
    metrics_dict: dictionary containing metric names (keys) and their corresponding 
                  scores (values), typically output from `evaluate_model()`

    [Returns]
    None: displays a bar plot showing accuracy, precision, recall, and F1-score.
    """
    plt.figure(figsize=(6,4))
    bars = plt.bar(metrics_dict.keys(), metrics_dict.values())
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Model Performance Metrics")
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.2f}", 
                 ha='center', va='bottom', fontsize=10)
        
    plt.show()

def plot_cv_accuracy(model, X, y, cv:int = 5, scoring:str = 'accuracy') -> None:
    """
    Plot the cross-validated accuracy scores of a model.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        The machine learning model to evaluate (e.g., LogisticRegression()).
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix.
    y : pandas.Series or numpy.ndarray
        Target vector.
    cv : int, default=5
        Number of cross-validation folds.
    scoring : str, default='accuracy'
        Scoring metric for evaluation.

    Returns
    -------
    None
        Displays a bar chart showing the cross-validation accuracy for each fold and the mean accuracy.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    # Plot CV scores
    plt.figure(figsize=(6,4))
    folds = [f"Fold {i+1}" for i in range(len(scores))]
    plt.bar(folds, scores, color='skyblue')
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title(f"{cv}-Fold Cross-Validated Accuracy")
    
    # Add value labels on top
    for i, score in enumerate(scores):
        plt.text(i, score + 0.01, f"{score:.2f}", ha='center', va='bottom', fontsize=10)
    
    # Plot mean line
    plt.axhline(scores.mean(), color='red', linestyle='--', label=f"Mean = {scores.mean():.2f}")
    plt.legend()
    plt.show()