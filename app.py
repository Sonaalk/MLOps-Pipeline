from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd
import logging


# Initialize app
app = FastAPI()


# Load trained model
model = mlflow.sklearn.load_model("models/final_model")


# Set up logging
logging.basicConfig(level=logging.INFO)


# Prediction endpoint
@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df).max()
    logging.info(f"Prediction request received: {data}")
    return {
        "prediction": int(prediction),
        "confidence": float(probability)
    }
