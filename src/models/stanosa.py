import torch
from torch import nn, optim
from einops import rearrange
from .linear_autoencoder import AutoEncoder
from torch.nn import functional as F
from collections import defaultdict
import mlflow


class StaNoSA(nn.Module):

    def __init__(self, **kwargs):
        """
        StaNoSA

        Parameters
        ----------
        kwargs: dict
            See argument for `linear_autoencoder.AutoEncoder`.
        """
        super(StaNoSA, self).__init__()

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

        # flatten patches into vectors
        x_flat = rearrange(x, "b p f w h -> (b p) (f w h)")

        # pass through autoencoder
        encoded, reconstructed = self.auto_encoder(x_flat)

        return encoded, reconstructed


def train_stanosa(
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda:0",
    epochs: int = 100,
    learning_rate: float = 0.0001
) -> StaNoSA:

    model = StaNoSA().to(torch.device(device))
    optim_stanosa = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):

        epoch_losses = defaultdict(list)

        for i, batch in enumerate(dataloader):
            batch = batch.to(torch.device(device))

            optim_stanosa.zero_grad()
            encoded, reconstructed = model(batch)
            loss, loss_dict = model.calculate_loss(batch, reconstructed, encoded)

            # update weights
            loss.backward()
            optim_stanosa.step()

            # save loss values
            for key, value in loss_dict.items():
                epoch_losses[key].append(value)

        for key, values in epoch_losses.items():
            avg = torch.FloatTensor(values).mean().item()
            mlflow.log_metric(key, avg, step=epoch)

    return model