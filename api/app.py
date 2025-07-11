# api/app.py
from fastapi import FastAPI, HTTPException
import pandas as pd
from src.predict import load_model_and_preprocessors, predict_credit_risk

app = FastAPI(title="Bank Credit Risk Prediction API")

@app.post("/predict/")
async def predict(data: dict):
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([data])
        
        # Predict
        predictions, probabilities = predict_credit_risk(input_data)
        
        return {
            "prediction": "Not Approved" if predictions[0] == 1 else "Approved",
            "default_probability": float(probabilities[0])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
