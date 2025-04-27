import pandas as pd
import numpy as np
import random

import warnings
warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler

import joblib

from sklearn.pipeline import Pipeline

class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature_to_drop=['Age']):
        self.feature_to_drop = feature_to_drop

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        if set(self.feature_to_drop).issubset(df.columns):
            df = df.drop(self.feature_to_drop, axis=1)
            return df
        else:
            print("One or more features are not in the dataframe")
            return df

# Handle skewness
class SkewnessHandler(BaseEstimator, TransformerMixin):
    def __init__(self, features=['Annual Income', 'Loan Amount']):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.features] = X[self.features].apply(np.cbrt)
        return X

# Min-Max scaling
class MinMaxScalerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, features=['Annual Income', 'Loan Amount', 'Loan To Income Ratio', 'Interest Rate', 'Employment Length', 'Credit History Length']):
        self.features = features
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.features])
        return self

    def transform(self, X):
        X[self.features] = self.scaler.transform(X[self.features])
        return X

# One-hot encoding
class OneHotEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, features=['Home Ownership', 'Loan Purpose']):
        self.features = features
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def fit(self, X, y=None):
        self.encoder.fit(X[self.features])
        return self

    def transform(self, X):
        encoded = self.encoder.transform(X[self.features])
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(self.features), index=X.index)
        X = X.drop(columns=self.features)
        return pd.concat([X, encoded_df], axis=1)

# Ordinal encoding
class OrdinalEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, features=['Credit Grade']):
        self.features = features
        self.encoder = OrdinalEncoder()

    def fit(self, X, y=None):
        self.encoder.fit(X[self.features])
        return self

    def transform(self, X):
        X[self.features] = self.encoder.transform(X[self.features])
        return X

# Binary mapping (e.g., Yes/No, Y/N)
class BinaryMapper(BaseEstimator, TransformerMixin):
    def __init__(self, mapping_features={'Has Prior Default': {'Y': 1, 'N': 0}}):
        self.mapping_features = mapping_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for feature, mapping in self.mapping_features.items():
            X[feature] = X[feature].map(mapping)
        return X

# Imputer for missing values
class ImputerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean', features=['Employment Length', 'Interest Rate']):
        self.strategy = strategy
        self.features = features

    def fit(self, X, y=None):
        if self.strategy == 'mean':
            self.fill_values = X[self.features].mean()
        elif self.strategy == 'median':
            self.fill_values = X[self.features].median()
        else:
            self.fill_values = X[self.features].mode().iloc[0]
        return self

    def transform(self, X):
        X[self.features] = X[self.features].fillna(self.fill_values)
        return X
        
def full_pipeline(df):
    pipeline = Pipeline([
        ('imputer', ImputerWrapper()),
        ('drop_features', DropFeatures()),
        ('skewness_handler', SkewnessHandler()),
        ('binary_mapper', BinaryMapper()),
        ('ordinal_encoder', OrdinalEncoderWrapper()),
        ('one_hot_encoder', OneHotEncoderWrapper()),
        ('min_max_scaler', MinMaxScalerWrapper())
    ])
    
    df_processed = pipeline.fit_transform(df)
    return df_processed

model_path = 'C:/Users/thanh/OneDrive/Desktop/New Project/Banking/saved_models/extra_trees/extra_trees_model.sav'
extra_trees_model = joblib.load(model_path)

def predict(data: pd.DataFrame):
    data = data.rename(columns={
        "person_age": "Age",
        "person_income": "Annual Income",
        "person_home_ownership": "Home Ownership",
        "person_emp_length": "Employment Length",
        "loan_intent": "Loan Purpose",
        "loan_grade": "Credit Grade",
        "loan_amnt": "Loan Amount",
        "loan_int_rate": "Interest Rate",
        "loan_status": "Loan Default",
        "loan_percent_income": "Loan To Income Ratio",
        "cb_person_default_on_file": "Has Prior Default",
        "cb_person_cred_hist_length": "Credit History Length"
    })

    data_processed = full_pipeline(data)

    X_data = data_processed.drop(columns='Loan Default')
    y_data = data_processed['Loan Default'].astype('int64')
    y_pred = extra_trees_model.predict(X_data)
    y_proba = extra_trees_model.predict_proba(X_data)
    
    result = data.copy()
    result['Predicted Default'] = y_pred
    result['Prob Non-Default'] = y_proba[:, 0]
    result['Prob Default'] = y_proba[:, 1]
    
    return result

