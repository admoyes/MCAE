import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from datasets import TripletPatchDataset
from models.mcae import train_mcae
import mlflow


# MCAE training
dataset = TripletPatchDataset(
    "/data",
    8
)
dataloader = DataLoader(
    dataset,
    num_workers=6,
    shuffle=True,
    batch_size=128,
)
with mlflow.start_run(experiment_id="mcae"):
    mcae_model = train_mcae(dataloader)