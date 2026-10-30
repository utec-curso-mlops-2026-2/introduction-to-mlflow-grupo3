import mlflow

def register_model(model, model_name, n_estimators, accuracy):

    mlflow.set_experiment("LightGBM Experiment")

    with mlflow.start_run():
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("model_type", "LightGBM")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, name="model")