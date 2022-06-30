import torch
from torch import nn, optim
from einops import rearrange
from .linear_autoencoder import AutoEncoder
from torch.nn import functional as F
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
        return None, None