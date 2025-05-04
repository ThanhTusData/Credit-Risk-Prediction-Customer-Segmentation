from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import json
import os
import sys
import joblib

# Add the current directory to sys.path to import predict.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler


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

def predict_churn(data):
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
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

    # Define paths relative to the current directory
    pipeline_path = os.path.join(current_dir, 'saved_models', 'pipeline', 'pipeline_preprocessing.sav')
    model_path = os.path.join(current_dir, 'saved_models', 'extra_trees', 'extra_trees_model.sav')
    
    pipeline = joblib.load(pipeline_path)

    df = data.copy()

    # Add Loan Default column if it's not present (for prediction mode)
    if 'Loan Default' not in df.columns:
        df['Loan Default'] = 'No'  # Default value, will be ignored during prediction

    data_processed = pipeline.transform(df)

    X_data_processed = data_processed.drop(columns='Loan Default')
    
    model = joblib.load(model_path)

    y_pred = model.predict(X_data_processed)
    y_proba = model.predict_proba(X_data_processed)

    # Add prediction results to original dataframe
    data['Predicted Default'] = y_pred
    data['Prob Non-Default'] = y_proba[:, 0]
    data['Prob Default'] = y_proba[:, 1]

    # # Map numerical predictions to Yes/No for better readability
    # data['Predicted Churn'] = data['Predicted Churn'].map({1: 'Yes', 0: 'No'})
    
    # # If Senior Citizen was numeric, map it back to Yes/No
    # if 'Senior Citizen' in data.columns and data['Senior Citizen'].dtype == 'int64':
    #     data['Senior Citizen'] = data['Senior Citizen'].map({1: 'Yes', 0: 'No'})

    return data

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        content_type = request.headers.get('Content-Type')
        
        if content_type == 'application/json':
            # Get JSON data from request
            json_data = request.json
            
            # Check if multiple customers or single customer
            if isinstance(json_data, list):
                # Multiple customers
                df = pd.DataFrame(json_data)
            else:
                # Single customer
                df = pd.DataFrame([json_data])
                
            # Make predictions using the predict_churn function
            result_df = predict_churn(df)
            
            # Convert DataFrame to dict for JSON response
            if len(result_df) == 1:
                # Return single customer result as object
                result = result_df.iloc[0].to_dict()
            else:
                # Return multiple customers result as array
                result = result_df.to_dict(orient='records')
            
            # Return the predictions
            return jsonify({
                'status': 'success',
                'predictions': result,
                'message': 'Churn prediction completed successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Unsupported Content-Type: {content_type}. Please send JSON data.'
            }), 400
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'API is running'
    })

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)