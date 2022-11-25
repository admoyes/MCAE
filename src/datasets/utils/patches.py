import torch
import numpy as np
from skimage.util import view_as_windows
from einops import rearrange


def get_patches(im: np.ndarray, patch_size: int) -> torch.FloatTensor:
    patches = torch.from_numpy(
        view_as_windows(
            im,
            (patch_size, patch_size, 3),
            step=(patch_size, patch_size, 3)
        )
    ).squeeze()

    # rearrange the patches
    patches = rearrange(
        patches, 
        "r c w h f -> (r c) f w h"
    )
    return patches.float()
