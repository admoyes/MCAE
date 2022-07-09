import mlflow
from mlflow.tracking.client import MlflowClient


def load_model(
    experiment_name: str, 
    run_name: str,
    model_url_prefix="/mlflow/projects/code/mlruns/",
    model_url_suffix="/artifacts/model/"
):
    """
    Load model from MLFlow artifact repository. 
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name("training")
    experiment_id = experiment.experiment_id
    run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        filter_string=f"tags.`mlflow.runName`='{run_name}'",
        max_results=1
    )[0]
    run_id = run.info.run_id
    model_url = model_url_prefix + experiment_id + "/" + run_id + model_url_suffix
    print("loading model from", model_url, flush=True)

    # load the model
    model = mlflow.pytorch.load_model(model_url).cpu()

    return model