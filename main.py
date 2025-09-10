import mlflow
import os
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

app = FastAPI(title = "Wate Potability Prediction", 
              version="1.0", 
              description="API for predicting water potability using a trained ML model.")

dagshub_url = "https://dagshub.com"
repo_owner = "ashu110081992"    
repo_name = "MLOps-with-GitActions"
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
mlflow.set_experiment("MLFLOW PIPELINE")


def load_model(model_name: str):
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(name=model_name, stages=["Staging"])
        run_id = versions[0].run_id
        model = mlflow.pyfunc.load_model(f"runs:/{run_id}/{model_name}")
        return model
    except Exception as e:
        raise Exception(f"Error loading model {model_name} from MLflow: {e}")
    
model = load_model("Best Model")

class WaterFeatures(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

@app.get("/")
def index():
    return {"message": "Welcome to the Water Potability Prediction API. Use the /predict endpoint to get predictions."}

@app.post("/predict")
def predict_potability(features: WaterFeatures):
    try:
        input_data = pd.DataFrame({
            'ph': [features.ph],
            'Hardness': [features.Hardness],
            'Solids': [features.Solids],
            'Chloramines': [features.Chloramines],
            'Sulfate': [features.Sulfate],
            'Conductivity': [features.Conductivity],
            'Organic_carbon': [features.Organic_carbon],
            'Trihalomethanes': [features.Trihalomethanes],
            'Turbidity': [features.Turbidity]
        })
        prediction = model.predict(input_data)
        potability = "Potable" if prediction[0] == 1 else "Not Potable"
        return {"potability": potability}
    except Exception as e:
        raise Exception(f"Error during prediction: {e}")