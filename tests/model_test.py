import pandas as pd
import os
import mlflow
import mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import unittest


# Initialize DagsHub for experiment tracking
# Initialize DagsHub for experiment tracking
#dagshub.init(repo_owner='ashu110081992', repo_name='MLOps-with-GitActions', mlflow=True)

MLOPS_TOKEN = os.getenv("MLOPS_TOKEN")
if not MLOPS_TOKEN:
    raise EnvironmentError("MLOPS_TOKEN environment variable not set.")

dagshub_url = "https://dagshub.com"
repo_owner = "ashu110081992"
repo_name = "MLOps-with-GitActions"

os.environ["MLFLOW_TRACKING_USERNAME"] = MLOPS_TOKEN
os.environ["MLFLOW_TRACKING_PASSWORD"] = MLOPS_TOKEN


mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
mlflow.set_experiment("MLFLOW PIPELINE")

model_name = "Best Model"

class TestModelLoading(unittest.TestCase):

    def test_model_in_staging(self):
        client = MlflowClient()
        versions = client.get_latest_versions(name=model_name, stages=["Staging"])
        self.assertGreater(len(versions), 0, f"No versions found for model")

    def test_model_loading(self):
        client = MlflowClient()
        versions = client.get_latest_versions(name=model_name, stages=["Staging"])
        if not versions:
            self.fail("No model versions found in Staging stage")

        latest_version = versions[0].version
        run_id = versions[0].run_id

        logged_model = f"runs:/{run_id}/{model_name}"

        try:
            loaded_model = mlflow.pyfunc.load_model(logged_model)
        except Exception as e:
            self.fail(f"Failed to load model: {e}")

        self.assertIsNotNone(loaded_model, "Loaded model is None")
        print(f"Model version {loaded_model} loaded successfully from Staging stage.")

if __name__ == "__main__":
    unittest.main()