import mlflow
from mlflow.tracking.client import MlflowClient
from math import sqrt
from einops import rearrange


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



def log_image(x, path, norm=True):
    """
    Log image to mlflow.

    Parameters
    ----------
    x: torch.FloatTensor
        image to save. Shape=[w, h, c] or [b, w, h, c].
        if batch of images, b must be square.
    """

    if x.shape[-1] != 3:
        raise ValueError(f"expected 3 channels but got {x.shape[-1]}")

    if len(x.shape) == 4:
        x = rearrange(
            x, 
            "(b1 b2) w h c -> (b1 w) (b2 h) c",
            b1=int(sqrt(x.shape[0]))
        )

    if norm:
        x += 1
        x /= 2
    
    x = x.detach().cpu().numpy()

    mlflow.log_image(x, path)