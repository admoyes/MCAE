from enum import auto
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from einops import rearrange
from .linear_autoencoder import AutoEncoder
from torch.nn import functional as F
from typing import Tuple
import mlflow
from collections import defaultdict
from .utils import load_model, log_image
from .classifiers import classifier_eval
from datasets import ClassificationDataset


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

        #print("MCAE")
        #print("x", x.shape, flush=True)

        if len(x.shape) == 5:
            # x doesn't have a domain dimension - add one in
            x = rearrange(x, "b p c w h -> b p 1 c w h")

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
        x_flat = rearrange(x, "b p d c w h -> (b p) d (c w h)")

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
        #print("calculate loss")
        #print("x", x.shape, "recon", reconstructed.shape, "encoded", encoded.shape)
        #print(f"[x] min: {x.min().item()} max: {x.max().item()}")
        #print(f"[reconstructed] min: {reconstructed.min().item()} max: {reconstructed.max().item()}", flush=True)

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
        optims.append(optim_ae)

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

            # save loss values
            for key, value in loss_dict.items():
                epoch_losses[key].append(value)

        for key, values in epoch_losses.items():
            avg = torch.FloatTensor(values).mean().item()
            mlflow.log_metric(key, avg, step=epoch)

        # log images
        batch_idx = torch.randperm(batch.shape[0])[:100]
        patch_idx = torch.randperm(batch.shape[1])[:100]
        batch_sample = batch[batch_idx, patch_idx].detach()
        for d, domain_batch_sample in enumerate(torch.unbind(batch_sample, 1)):
            domain_batch_sample = rearrange(domain_batch_sample, "b f w h -> b w h f")
            log_image(domain_batch_sample, f"domain_{d}_batch.png")

        recon_sample = reconstructed[batch_idx, patch_idx].detach()
        for d, domain_recon_sample in enumerate(torch.unbind(recon_sample, 1)):
            domain_recon_sample = rearrange(domain_recon_sample, "b f w h -> b w h f")
            log_image(domain_recon_sample, f"domain_{d}_recon.png")

    # log model to mlflow
    mlflow.pytorch.log_model(model, "model")

    # clear GPU
    #torch.cuda.empty_cache()



def eval_mcae(
    data_path: str,
    classifier_name: str,
    training_experiment_name: str,
    training_run_name: str,
    batch_size: int,
    patch_size: int,
    domain_index: int,
    num_samples: int = 10
):

    # load model from mlflow
    model = load_model(training_experiment_name, training_run_name)

    # define the classification dataset and dataloader
    lung_dataset = ClassificationDataset(
        image_folder=data_path,
        patch_size=patch_size,
        image_file_extension="jpeg",
        autoencoder=model,
        patch_transform=None,
        model_call_args={
            "index": torch.zeros(1).fill_(domain_index).long()
        }
    )
    
    dataloader = DataLoader(lung_dataset, batch_size=batch_size, shuffle=True)

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