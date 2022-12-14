import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from einops import rearrange
from .linear_autoencoder import AutoEncoder
from datasets.utils.ZCA import compute_zca_matrix, apply_zca_to_batch
from datasets.utils.sample import sample_from_dataloader
from datasets.utils.norm import global_contrast_normalisation
from datasets import BasicPatchDataset, ClassificationDataset
from torch.nn import functional as F
from collections import defaultdict
import mlflow
from typing import Tuple
from tqdm import tqdm
from .utils import load_model
from .classifiers import classifier_eval


class StaNoSA(nn.Module):

    def __init__(self, zca_mean: torch.FloatTensor, zca_components: torch.FloatTensor, **kwargs):
        """
        StaNoSA

        Parameters
        ----------
        zca_mean: torch.FloatTensor
            The precomputed sample mean used for ZCA whitening.
        zca_components: torch.FloatTensor
            The precomputed ZCA transformation components.
        kwargs: dict
            See argument for `linear_autoencoder.AutoEncoder`.
        """
        super(StaNoSA, self).__init__()
        self.zca_mean = zca_mean
        self.zca_components = zca_components

        # define auto encoder
        self.auto_encoder = AutoEncoder(**kwargs)


    def forward(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Run sub-patches through the autoencoder. Assume that inputs are already ZCA whitened.

        Parameters
        ----------
        x: torch.FloatTensor
            Tensor of input patches. [batch size, patches, features, width, height].\
            Note, `features` refers to colour channels.
        """

        # get dimensions
        batch_size, num_patches, num_features, width, height = x.shape

        # flatten patches into vectors
        x_flat = rearrange(x, "b p f w h -> (b p) (f w h)")

        # apply ZCA whitening
        x_flat = apply_zca_to_batch(self.zca_mean, self.zca_components, x_flat) 

        # pass through autoencoder
        encoded, reconstructed = self.auto_encoder(x_flat)

        # rearrange reconstructed back into patch form
        reconstructed = rearrange(
            reconstructed,
            "(b p) (f w h) -> b p f w h",
            b=batch_size, p=num_patches,
            f=num_features, w=width, h=height
        )

        return encoded, reconstructed

    def calculate_loss(self, x, reconstructed, encoded) -> Tuple[torch.FloatTensor, dict]:

        mse_loss = F.mse_loss(reconstructed, x)

        return mse_loss, {
            "reconstruction": mse_loss
        }


def train_stanosa(
    dataloader: torch.utils.data.DataLoader,
    domain_index: int, 
    device: str = "cuda:0",
    epochs: int = 100,
    learning_rate: float = 0.0001,
    zca_n_samples: int = 50_000
) -> StaNoSA:

    run = mlflow.active_run()
    print(f"StaNoSA Training")

    # compute ZCA components
    sample = sample_from_dataloader(dataloader, domain_index, zca_n_samples)
    zca_mean, zca_components = compute_zca_matrix(sample)
    # move zca objects onto correct device
    zca_mean = zca_mean.to(torch.device(device))
    zca_components = zca_components.to(torch.device(device))

    # define model and optimiser
    model = StaNoSA(zca_mean, zca_components).to(torch.device(device))
    optim_stanosa = optim.Adam(model.parameters(), lr=learning_rate)

    # start training loop
    for epoch in tqdm(range(epochs), total=epochs, desc="epoch"):

        epoch_losses = defaultdict(list)

        for i, batch in enumerate(dataloader, 0):
            # select the correct domain from the batch
            domain_batch = batch[:, :, domain_index, ...]

            # move to correct device
            domain_batch = domain_batch.to(torch.device(device))

            # forward pass
            optim_stanosa.zero_grad()
            encoded, reconstructed = model(domain_batch)
            loss, loss_dict = model.calculate_loss(domain_batch, reconstructed, encoded)

            # update weights
            loss.backward()
            optim_stanosa.step()

            # save loss values
            for key, value in loss_dict.items():
                epoch_losses[key].append(value)

        for key, values in epoch_losses.items():
            avg = torch.FloatTensor(values).mean().item()
            mlflow.log_metric(key, avg, step=epoch)

    # log model to mlflow
    mlflow.pytorch.log_model(model, "model")


def eval_stanosa(
    data_path: str, 
    classifier_name: str,
    training_experiment_name: str,
    training_run_name: str,
    batch_size: int,
    patch_size: int,
    n_samples=10,
):

    # ZCA components need to be calculated for the basic classification dataset
    # start by sampling a number of patches from the dataset
    print("building BasicPatchDataset")
    patch_dataset = BasicPatchDataset(
        image_folder=data_path,
        patch_size=patch_size,
        image_file_extension="jpeg",
        patch_transform=tf.Compose([
            tf.Lambda(lambda patches: global_contrast_normalisation(patches)),
            tf.Lambda(lambda patches: rearrange(patches, "p f w h -> p 1 f w h"))
        ])
    )
    patch_dataloader = DataLoader(
        patch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6
    )

    # sample patches from the dataloader
    samples = []
    with tqdm(total=n_samples, desc="samples for ZCA") as prog:
        for batch in patch_dataloader:
            batch = batch.squeeze()
            batch_flat = rearrange(batch, "b p f w h -> (b p) (f w h)")
            idx = torch.randperm(len(batch_flat))
            batch_sample = batch_flat[idx][:100]
            samples.append(batch_sample)
            prog.update(len(batch_sample))
            if len(samples) >= n_samples:
                break
    samples = torch.cat(samples, dim=0)

    # compute ZCA components
    zca_mean, zca_components = compute_zca_matrix(samples)

    # load model from mlflow
    model = load_model(training_experiment_name, training_run_name)

    # set the ZCA parameters
    model.zca_mean = zca_mean
    model.zca_components = zca_components

    # define the classification dataset and dataloader 
    lung_dataset = ClassificationDataset(
        image_folder=data_path,
        patch_size=patch_size,
        image_file_extension="jpeg",
        autoencoder=model,
        patch_transform=tf.Lambda(lambda patches: global_contrast_normalisation(patches))
    )

    dataloader = DataLoader(lung_dataset, batch_size=batch_size, shuffle=True, num_workers=6)

    # load all the features and labels into memory
    all_features = []
    all_labels = []
    for x, y in dataloader:
        all_features.append(x)
        all_labels.append(y)
    
    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # perform eval with classifier
    classifier_eval(classifier_name, all_features, all_labels)
