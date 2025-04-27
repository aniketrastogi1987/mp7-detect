from typing import List
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin



class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        """
        Initialize the handler with optional columns and a factor for IQR.

        :param columns: List of columns to apply the outlier handling. If None, apply to all numerical columns.
        :param factor: The factor to multiply with IQR to determine bounds. Default is 1.5.
        """
        self.columns = columns
        
    def fit(self, X, y=None):
        # Determine which columns to process
        self.q1 = {col: np.percentile(X[col], 25) for col in self.columns}
        self.q3 = {col: np.percentile(X[col], 75) for col in self.columns}
        self.iqr = {col: self.q3[col] - self.q1[col] for col in self.columns}
        return self
        #columns_to_process = self.columns if self.columns is not None else X.select_dtypes(include=[np.number]).columns

        # Calculate bounds for each column
        #for column in columns_to_process:
        #    Q1 = X[column].quantile(0.25)
        #    Q3 = X[column].quantile(0.75)
        #    IQR = Q3 - Q1
        #    lower_bound = Q1 - self.factor * IQR
        #    upper_bound = Q3 + self.factor * IQR
        #    self.bounds[column] = {'lower': lower_bound, 'upper': upper_bound}

        #return self
    def transform(self, X):
        """Handle outliers in the data."""
        X_copy = X.copy()
        for col in self.columns:
            col_dtype = X[col].dtype
            lower_bound = self.q1[col] - 1.5 * self.iqr[col]
            upper_bound = self.q3[col] + 1.5 * self.iqr[col]
            
            # Cast bounds to match column dtype
            lower_bound = lower_bound.astype(col_dtype)
            upper_bound = upper_bound.astype(col_dtype)

            # Replace outliers with bounds
            mask_lower = X_copy[col] < lower_bound
            mask_upper = X_copy[col] > upper_bound

            # Replace outliers with bounds
            X_copy.loc[X_copy[col] < lower_bound, col] = lower_bound
            X_copy.loc[X_copy[col] > upper_bound, col] = upper_bound
            
        return X_copy