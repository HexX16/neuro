from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class DomainScaler(BaseEstimator, TransformerMixin):
    def __init__(self, per_feature_min, per_feature_max, feature_range=(0, 5)):
        self.per_feature_min = per_feature_min
        self.per_feature_max = per_feature_max
        self.feature_range = feature_range
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = list(range(X.shape[1]))
        return self

    def transform(self, X):
        a, b = self.feature_range
        if isinstance(X, pd.DataFrame):
            X_out = X.copy()
        else:
            X_out = pd.DataFrame(X, columns=self.feature_names_in_)

        for col in self.feature_names_in_:
    
            min_val = self.per_feature_min[col]
            max_val = self.per_feature_max[col]

            X_out[col] = np.clip(X_out[col], min_val, max_val)

            if (X_out[col] < min_val).any() or (X_out[col] > max_val).any():
                print(f"Warning: column {col} has values outside the defined range!")
                
            X_out[col] = X_out[col].fillna(min_val)

            if max_val == min_val:
                X_out[col] = a  # если нет диапазона
            else:
                X_out[col] = (X_out[col] - min_val) / (max_val - min_val) * (b - a) + a

        return X_out
