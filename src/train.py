import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from datasets import TripletPatchDataset
from models.mcae import train_mcae
from models.stanosa import train_stanosa
import mlflow

# define stanosa dataset and dataloader
dataset = TripletPatchDataset("/data", 8, False)

dataloader = DataLoader(
    dataset,
    num_workers=6,
    shuffle=True,
    batch_size=128,
)

# train stanosa models
with mlflow.start_run(run_name="stanosa_a", description="StaNoSA Training - Domain A"):
    mlflow.set_tag("mlflow.runName", "stanosa_a")
    train_stanosa(dataloader, 0)

with mlflow.start_run(run_name="stanosa_b", description="StaNoSA Training - Domain B"):
    mlflow.set_tag("mlflow.runName", "stanosa_b")
    train_stanosa(dataloader, 1)

with mlflow.start_run(run_name="stanosa_c", description="StaNoSA Training = Domain C"):
    mlflow.set_tag("mlflow.runName", "stanosa_c")
    train_stanosa(dataloader, 2)

# define the MCAE dataset and dataloader
dataset = TripletPatchDataset("/data", 8, True)

dataloader = DataLoader(
    dataset,
    num_workers=6,
    shuffle=True,
    batch_size=128,
)

# train MCAE model
with mlflow.start_run(run_name="MCAE", description="MCAE Training"):
    train_mcae(dataloader)