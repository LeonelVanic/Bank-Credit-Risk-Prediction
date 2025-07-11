# Bank Credit Risk Prediction

A machine learning project to predict whether a bank client is likely to default on a loan, helping banks make informed credit approval decisions.

## Overview
This project uses the UCI Credit Approval Dataset to train a Random Forest model for binary classification (default vs. no default). It includes data preprocessing, model training, evaluation, and a FastAPI endpoint for real-time predictions.

## Features
- **Data Preprocessing**: Handles missing values, encodes categorical features, and scales numerical features.
- **Model**: Random Forest classifier with AUC-ROC optimization.
- **API**: FastAPI endpoint for serving predictions.
- **EDA**: Jupyter Notebook for exploratory data analysis.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Bank-Credit-Risk-Prediction.git
   cd Bank-Credit-Risk-Prediction
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Download the UCI Credit Approval Dataset (`crx.data`) and place it in `data/raw/`.

## Usage
1. **Run Preprocessing and Training**:
   ```bash
   python src/train_model.py
   ```
2. **Make Predictions**:
   ```bash
   python src/predict.py
   ```
3. **Start the API**:
   ```bash
   uvicorn api.app:app --host 0.0.0.0 --port 8000
   ```
   Test the API with a POST request to `http://localhost:8000/predict/` with a JSON payload like:
   ```json
   {
       "A1": "b", "A2": 30.83, "A3": 0, "A4": "u", "A5": "g",
       "A6": "w", "A7": "v", "A8": 1.25, "A9": "t", "A10": "t",
       "A11": 1, "A12": "f", "A13": "g", "A14": 202, "A15": 0
   }
   ```

## Project Structure
- `data/`: Raw and processed datasets.
- `src/`: Python scripts for preprocessing, training, and prediction.
- `api/`: FastAPI application for deployment.
- `notebooks/`: Jupyter Notebook for EDA.
- `models/`: Trained model and preprocessing objects.

## License
MIT License
