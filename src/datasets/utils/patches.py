import torch
import numpy as np
from skimage.util import view_as_windows


def get_patches(im: np.ndarray, patch_size: int) -> torch.FloatTensor:
    patches = torch.from_numpy(
        view_as_windows(
            im,
            (patch_size, patch_size, 3),
            step=(patch_size, patch_size, 3)
        )
    )
    return patches.contiguous().view(-1, patch_size, patch_size, 3).permute(0, 3, 1, 2).contiguous()
