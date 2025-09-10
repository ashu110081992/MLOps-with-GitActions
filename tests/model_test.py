import pandas as pd
import os
import mlflow
from mlflow.tracking import MlflowClient
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


    def test_model_performance(self):
        client = MlflowClient()
        versions = client.get_latest_versions(name=model_name, stages=["Staging"])
        if not versions:
            self.fail("No model versions found in Staging stage")

        latest_version = versions[0].version
        run_id = versions[0].run_id
        logged_model = f"runs:/{latest_version}/{model_name}"

        try:
            loaded_model = mlflow.pyfunc.load_model(logged_model)
        except Exception as e:
            self.fail(f"Failed to load model: {e}")

        self.assertIsNotNone(loaded_model, "Loaded model is None")

        test_data_path = os.path.join("data", "processed", "test_processed.csv")
        try:
            test_data = pd.read_csv(test_data_path)
        except Exception as e:
            self.fail(f"Failed to load test data: {e}")

        if 'Potability' not in test_data.columns:
            self.fail("Test data does not contain 'Potability' column")

        X_test = test_data.drop(columns=['Potability'], axis=1)
        y_test = test_data['Potability']

        try:
            y_pred = loaded_model.predict(X_test)
        except Exception as e:
            self.fail(f"Failed to make predictions: {e}")

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Model Performance on Test Data:")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

        self.assertGreaterEqual(accuracy, 0.3, "Model accuracy is below acceptable threshold")
        self.assertGreaterEqual(precision, 0.3, "Model precision is below acceptable threshold")
        self.assertGreaterEqual(recall, 0.3, "Model recall is below acceptable threshold")
        self.assertGreaterEqual(f1, 0.3, "Model F1 score is below acceptable threshold")

if __name__ == "__main__":
    unittest.main()