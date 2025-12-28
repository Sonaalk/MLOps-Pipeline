import mlflow.sklearn
import pandas as pd
import os

# Check if trained model loads correctly
def test_model_load():
    model_path = os.path.join("models", "final_model")
    model = mlflow.sklearn.load_model(model_path)
    assert model is not None

# Check if model can make a prediction
def test_model_prediction():
    model_path = os.path.join("models", "final_model")
    model = mlflow.sklearn.load_model(model_path)

    # Sample input (single row)
    sample = {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1
    }

    df = pd.DataFrame([sample])
    prediction = model.predict(df)

    assert prediction[0] in [0, 1]
