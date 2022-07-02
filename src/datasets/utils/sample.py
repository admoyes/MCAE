import torch
from torch.utils.data import DataLoader
from einops import rearrange


def sample_from_dataloader(
    dataloader: DataLoader,
    domain_index: int,
    num_samples: int,
    samples_per_batch: int = 250
) -> torch.FloatTensor:
    """
    Take samples of examples from a multi-domain dataloader.

    Parameters
    ----------
    dataloader: DataLoader
        The multi-domain dataloader to sample from.
    domain_index: int
        Specifies which domain to select from each batch.
    num_samples: int
        How many samples to take in total.
    samples_per_batch: int
        How many samples will be taken from each batch

    Returns
    -------
    torch.FloatTensor
        The sampled data. Shape=[num_samples, num_features]
    """

    sample = []
    for i, batch in enumerate(dataloader, 0):
        # batch shape=[batch_size, num_patches, num_domains, num_features, width, height]
        # select the correct domain
        batch_domain = batch[:, :, domain_index, ...]

        # flatten batch
        batch_flat = rearrange(batch_domain, "b p f w h -> (b p) (f w h)")

        # shuffle and take sample
        idx = torch.randperm(len(batch_flat))
        batch_sample = batch_flat[idx][:samples_per_batch]

        # save sample
        sample.append(batch_sample)

    # concatenate all samples into a single matrix
    sample = torch.cat(sample, dim=0)

    # shuffle and take sample
    idx = torch.randperm(len(sample))
    sample = sample[idx][:num_samples]

    return sample
