import torch
from torch import nn
from typing import Tuple



class AutoEncoder(nn.Module):

    def __init__(self, input_dim=192, hidden_dim=100, output_dim=10, hidden_activation=nn.Tanh, output_activation=nn.Tanh, hidden_dropout=0.2) -> None:
        """Basic linear auto encoder class.

        Parameters
        ----------
        input_dim: int
            Size of the input vectors.
        hidden_dim: int
            Number of neurons in the hidden layer.
        output_dim: int
            Number of neurons on the output layer (should be the same as input_dim).
        hidden_activation:
            Activation function to use for the hidden layer. Can be None.
        output_activation:
            Activation function to use for the output layer. Can be None.
        """
        super(AutoEncoder, self).__init__()

        # define encoder
        hidden_activation_fn = hidden_activation() if hidden_activation is not None else nn.Identity()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            hidden_activation_fn,
            nn.Linear(hidden_dim, output_dim),
            hidden_activation_fn,
            nn.Dropout(hidden_dropout) if hidden_dropout > 0.0 else nn.Identity(),
        )

        # define decoder
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            hidden_activation_fn,
            nn.Linear(hidden_dim, input_dim),
            output_activation() if output_activation is not None else nn.Identity()
        )


    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x) # encoded representation
        x_hat = self.decoder(z) # reconstruction of input
        return z, x_hat
        

