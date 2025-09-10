# import mlflow
# import dagshub

# mlflow.set_tracking_uri("https://dagshub.com/ashu110081992/MLOps-with-GitActions.mlflow")

# dagshub.init(repo_owner='ashu110081992', repo_name='MLOps-with-GitActions', mlflow=True)

# with mlflow.start_run():
#   mlflow.log_param('parameter name', 'value')
#   mlflow.log_metric('metric name', 1)
import pandas as pd

data_filepath = "https://raw.githubusercontent.com/ashu110081992/Dataset/main/water_potability.csv"

df = pd.read_csv(data_filepath, index_col=0)
print(df.head())