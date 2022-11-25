import torch
from torch.utils.data import DataLoader
import numpy as np
from typing import Union, Tuple


def compute_zca_matrix(x: Union[torch.FloatTensor, np.ndarray], regularisation=1e-4) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """
    Calculate the components required to ZCA whiten the data.

    Parameters
    ----------
    x: Union[torch.FloatTensor, np.ndarray]
        Input data. Shape=[num_examples, num_features].

    Returns
    -------
    torch.FloatTensor
        Calculate mean. Use this to zero-centre data. Shape=[1, num_features]
    torch.FloatTensor
        ZCA transformation matrix. Shape=[num_features, num_features]
    """

    # convert to numpy
    if not isinstance(x, np.ndarray):
        x = x.numpy()

    # zero-centre x
    mean = x.mean(axis=0).reshape((1, -1)) # [1, num_features] 
    x = x - mean

    # variances
    sigma = np.dot(x.T, x) / (len(x) - 1)

    # svd
    U, S, V = np.linalg.svd(sigma)

    # calculate transformation matrix
    temp = np.dot(
        U,
        np.diag(1 / np.sqrt(S + regularisation))
    )
    components = np.dot(temp, U.T)

    return torch.from_numpy(mean), torch.from_numpy(components)


def apply_zca_to_batch(mean: torch.FloatTensor, components: torch.FloatTensor, x: torch.FloatTensor) -> torch.FloatTensor:
    """
    Apply ZCA whitening to a batch of data.

    Parameters
    ----------
    mean: torch.FloatTensor
        The precomputed sample mean. Shape=[1, num_features]
    components: torch.FloatTensor
        The precomputed ZCA transformation matrix. Shape=[num_features, num_features]
    x: torch.FloatTensor
        A batch of data. Shape=[num_examples, num_features]
    """

    # zero-centre x
    x = x - mean

    # whiten x
    #x = torch.dot(x, components.T)
    x = x @ components.T

    return x