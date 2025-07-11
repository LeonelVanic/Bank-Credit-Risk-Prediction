# src/train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import joblib
from data_preprocessing import load_data, preprocess_data

def train_model(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("AUC-ROC:", roc_auc_score(y_test, y_pred_proba))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    print("Cross-Validation AUC-ROC:", np.mean(cv_scores))
    
    return model

if __name__ == "__main__":
    # Load and preprocess data
    data = load_data("../data/raw/crx.data")
    X, y, preprocessor, scaler = preprocess_data(data)
    
    # Train model
    model = train_model(X, y)
    
    # Save model and preprocessing objects
    joblib.dump(model, "../models/credit_risk_model.pkl")
    joblib.dump(preprocessor, "../models/preprocessor.pkl")
    joblib.dump(scaler, "../models/scaler.pkl")
    print("Model and preprocessors saved!")
