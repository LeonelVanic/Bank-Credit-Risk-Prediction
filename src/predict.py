# src/predict.py
import pandas as pd
import joblib

def load_model_and_preprocessors():
    model = joblib.load("../models/credit_risk_model.pkl")
    preprocessor = joblib.load("../models/preprocessor.pkl")
    scaler = joblib.load("../models/scaler.pkl")
    return model, preprocessor, scaler

def predict_credit_risk(new_data):
    model, preprocessor, scaler = load_model_and_preprocessors()
    
    # Preprocess new data
    X_processed = preprocessor.transform(new_data)
    X_scaled = scaler.transform(X_processed)
    
    # Predict
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    return predictions, probabilities

if __name__ == "__main__":
    # Example: Load a single sample for testing
    sample_data = pd.DataFrame({
        'A1': ['b'], 'A2': [30.83], 'A3': [0], 'A4': ['u'], 'A5': ['g'], 
        'A6': ['w'], 'A7': ['v'], 'A8': [1.25], 'A9': ['t'], 'A10': ['t'], 
        'A11': [1], 'A12': ['f'], 'A13': ['g'], 'A14': [202], 'A15': [0]
    })
    predictions, probabilities = predict_credit_risk(sample_data)
    print("Prediction (0=Approved, 1=Not Approved):", predictions)
    print("Default Probability:", probabilities)
