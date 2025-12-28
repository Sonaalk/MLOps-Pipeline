from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd

# Initialize app
app = FastAPI()

# Load trained model
model = mlflow.sklearn.load_model("models/final_model")

# Prediction endpoint
@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df).max()
    
    return {
        "prediction": int(prediction),
        "confidence": float(probability)
    }
