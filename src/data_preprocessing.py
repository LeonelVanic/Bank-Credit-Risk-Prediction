# src/data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

def load_data(file_path):
    # Load UCI Credit Approval Dataset (update column names based on dataset)
    column_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 
                    'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'target']
    data = pd.read_csv(file_path, names=column_names)
    return data

def preprocess_data(data):
    # Replace '?' with NaN
    data = data.replace('?', np.nan)
    
    # Define feature types
    categorical_features = ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13']
    numerical_features = ['A2', 'A3', 'A8', 'A11', 'A14', 'A15']
    
    # Split features and target
    X = data.drop('target', axis=1)
    y = data['target'].map({'+': 1, '-': 0})  # Convert target to binary (1=default, 0=no default)
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), numerical_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ])
    
    # Fit and transform
    X_processed = preprocessor.fit_transform(X)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_scaled, y)
    
    return X_balanced, y_balanced, preprocessor, scaler

if __name__ == "__main__":
    data = load_data("../data/raw/crx.data")
    X, y, preprocessor, scaler = preprocess_data(data)
    print("Data preprocessed successfully!")
