# Pandas Notes

## Concepts

`Downcasting`: changing the data type of a column to a smaller or more memory-efficient type without losing information

## Cheatsheet

```python
""" Attributes """
df.shape        # (rows, columns)
df.columns      # column names

""" Methods """
df.head(n)       # first n rows (default=5)
df.tail(n)       # last 5 rows (default=5)
df.info()       # column info & dtypes
df.describe()   # summary statistics

df.to_csv()     # save to .csv
pd.read_csv(
    filepath_or_buffer,  # Path to the CSV file (string) or URL
    sep=',',             # Delimiter (default is comma)
    header='infer',      # Row to use as column names (default first row)
    names=None,          # List of column names to use
    index_col=None,      # Column(s) to use as index
    usecols=None,        # Subset of columns to load
    dtype=None,          # Data types for columns
    skiprows=None,       # Rows to skip at start
    nrows=None,          # Number of rows to read
    na_values=None,      # Additional strings to recognize as NaN
    parse_dates=False,   # Parse date columns
    infer_datetime_format=False,  # Try to infer datetime format

    df.astype('category') # Downcast categories
    pd.to_numeric(df, downcast='') # Downcast numericals

)
```

## Additional Notes

1. 
