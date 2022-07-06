from enum import auto
import torch
from torch import nn, optim
from einops import rearrange
from .linear_autoencoder import AutoEncoder
from torch.nn import functional as F
from typing import Tuple
import mlflow
from collections import defaultdict


class MultiChannelAutoEncoderLinear(nn.Module):

    def __init__(self, num_channels: int, **kwargs):
        """
        Multi-Channel AutoEncoder

        Parameters
        ----------
        num_channels: int
            The number of channels/domains (3 in paper). **This is not the number of features**.
        kwargs: dict
            See arguments for `linear_autoencoder.AutoEncoder`.
        """
        super(MultiChannelAutoEncoderLinear, self).__init__()

        # define auto encoders for each channel/domain
        self.auto_encoders = nn.ModuleList([AutoEncoder(**kwargs) for _ in range(num_channels)])

    def forward(self, x: torch.FloatTensor, index: torch.LongTensor = None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Run inputs through each of the auto encoders.

        Inputs are assigned to auto encoders based on the order they are passed in.
        E.g. given x with shape [batch size, 3, ...], the batch of patches for the first domain
        i.e. x[:, 0, ...] will be passed to `self.auto_encoders[0]`, the batch of patches for
        the second domain x[:, 1, ...] will be passed to `self.auto_encoders[1]` and so on.

        Note to self:
        It may be useful to include a parameter that allows some given data to be passed directly
        to a specific auto encoder rather than relying on order. This would be helpful for 
        inference/eval.

        Parameters
        ----------
        x: torch.FloatTensor:
            Tensor of input patches. [batch size, channels, patches, features, width, height]
        index: torch.LongTensor:
            Used to route domains to autoencoders. E.g. passing index=[2, 0, 1] will route the first domain x[:, 0, ...] to the 2nd autoencoder, the second domain to the 0th autoencoder, etc.

        Returns
        -------
        z: torch.FloatTensor:
            Tensor of hidden activations. [batch size, channels, patches, features]
        x_hat: torch.FloatTensor:
            Tensor of reconstructed patches. [batch size, channels, patches, features, width, height]
        """

        # deconstruct tensor shape
        batch_size, n_patches, n_domains, n_colour_channels, patch_width, patch_height = x.shape

        # check if index has been provided
        if index is None:

            # make sure the correct number of domains have been provided
            n_autoencoders = len(self.auto_encoders)
            if n_domains != n_autoencoders:
                raise ValueError(f"n_domains ({n_domains}) != n_autoencoders ({n_autoencoders}). either provide a corresponding number of domains or provide the `index` arg.")
            else:
                # otherwise create the default value
                # this will route domains to autoencoders in the standard order-based fashion.
                index = torch.arange(n_domains).long()
        else:
            # make sure the length of `index` is equal to `n_domains`
            if len(index) != n_domains:
                raise ValueError(f"len(index) ({len(index)}) != n_domains ({n_domains})")

            index = index.long()

        # flatten the patches into vectors
        x_flat = rearrange(x, "b p d c h w -> (b p) d (c h w)")

        # unbind x along the `channels` dimension so we go from a single tensor to a list of tensors
        domains = torch.unbind(x_flat, 1)

        # pass each of the inputs through the corresponding auto encoder
        all_encoded = []
        all_reconstructed = []
        for domain_x, i in zip(domains, index):
            i = i.item() # convert to an int
            encoded, reconstructed = self.auto_encoders[i](domain_x)
            all_encoded.append(encoded)
            all_reconstructed.append(reconstructed)

        # restack the activations and reconstructions into single tensors
        all_encoded = torch.stack(all_encoded, dim=1)
        encoded_dim = all_encoded.shape[-1]
        all_encoded = rearrange(all_encoded, "(b p) d c -> b p d c", b=batch_size, p=n_patches, d=n_domains, c=encoded_dim)

        all_reconstructed = torch.stack(all_reconstructed, dim=1)
        all_reconstructed = rearrange(all_reconstructed, "(b p) d (c w h) -> b p d c w h", b=batch_size, p=n_patches, d=n_domains, c=3, w=patch_width, h=patch_height)

        # return
        return all_encoded, all_reconstructed

    def calculate_loss(self, x, reconstructed, encoded) -> Tuple[torch.FloatTensor, dict]:

        mse_loss = F.mse_loss(reconstructed, x)
        feature_activation_loss = 0.01 * encoded.pow(2).mean()

        # calculate feature loss
        encoded_flat = rearrange(encoded, "b p d l -> (b p l) d")
        variance = encoded_flat.var(dim=1).mean()

        # combine losses
        loss = mse_loss + feature_activation_loss + variance

        return loss, {
            "reconstruction": mse_loss.detach().item(),
            "feature_magnitude": feature_activation_loss.detach().item(),
            "feature_variation": variance.detach().item()
        } 

def train_mcae(
    dataloader: torch.utils.data.DataLoader,
    num_domains: int = 3,
    device: str = "cuda:0",
    epochs: int = 100,
    learning_rate: float = 0.0001
) -> MultiChannelAutoEncoderLinear:

    run = mlflow.active_run()
    print(f"MCAE/DCAE Training")

    model = MultiChannelAutoEncoderLinear(num_domains).to(torch.device(device))
    # build optimisers
    optims = []
    for auto_encoder in model.auto_encoders:
        optim_ae = optim.Adam(auto_encoder.parameters(), lr=learning_rate)
    #optim_mcae = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):

        epoch_losses = defaultdict(list)

        for i, batch in enumerate(dataloader, 0):
            batch = batch.to(torch.device(device))

            for opt in optims:
                opt.zero_grad()

            encoded, reconstructed = model(batch)
            loss, loss_dict = model.calculate_loss(batch, reconstructed, encoded)
            
            # update weights
            loss.backward()

            for opt in optims:
                opt.step()
            #optim_mcae.step()

            # save loss values
            for key, value in loss_dict.items():
                epoch_losses[key].append(value)

        for key, values in epoch_losses.items():
            avg = torch.FloatTensor(values).mean().item()
            mlflow.log_metric(key, avg, step=epoch)

    # log model to mlflow
    mlflow.pytorch.log_model(model, "model")

    # clear GPU
    #torch.cuda.empty_cache()