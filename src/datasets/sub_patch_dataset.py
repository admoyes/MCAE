import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from skimage.util import view_as_windows


class TripletPatchDataset(Dataset):

    def __init__(self, root_path: str, patch_size: int, norm: bool, transform=None) -> None:
        """
        Dataset derived from the CycleGAN-generated dataset.
        Returns corresponding triplets of image sub-patches.

        Parameters
        ----------
        root_path: str
            Root path of the dataset. Directories under this path should be \
            "/A", "/B" and "/C".
        patch_size: int
            The size of each sub-patch to be extracted. (8 is probably a good choice).
        """
        self.root_path = Path(root_path)
        self.patch_size = patch_size
        self.norm = norm
        self.transform = transform

        # build up paths to sub-directories for each domain
        A_dir = self.root_path / "A"
        B_dir = self.root_path / "B"
        C_dir = self.root_path / "C"

        # find out how many images there are in the first domain
        # note, each domain should have the same number of patches
        num_images = len(list(A_dir.iterdir()))

        # build up paths to the images from each domain
        self.paths = []
        for i in range(1, num_images + 1):
            fn = f"{i}.png"
            self.paths.append((
                A_dir / fn,
                B_dir / fn,
                C_dir / fn
            ))
        
    def __len__(self) -> int:
        return len(self.paths)

    def get_patches(self, im) -> torch.FloatTensor:
        patches = torch.from_numpy(view_as_windows(im, (self.patch_size, self.patch_size, 3), step=(self.patch_size, self.patch_size, 3)))
        return patches.contiguous().view(-1, self.patch_size, self.patch_size, 3).permute(0, 3, 1, 2).contiguous()

    def load_image_as_patches(self, path: str) -> torch.FloatTensor:
        im = np.array(Image.open(path))
        patches = self.get_patches(im)
        return patches

    @staticmethod
    def normalise(tensor):
        mean = torch.as_tensor((0.5, 0.5, 0.5), dtype=tensor.dtype, device=tensor.device).view(1, 1, 3, 1, 1)
        std = torch.as_tensor((0.5, 0.5, 0.5), dtype=tensor.dtype, device=tensor.device).view(1, 1, 3, 1, 1)
        return tensor.sub_(mean).div_(std)

    def __getitem__(self, index):
        # extract paths
        a_path, b_path, c_path = self.paths[index]

        # extract patches
        patches_a = self.load_image_as_patches(a_path)
        patches_b = self.load_image_as_patches(b_path)
        patches_c = self.load_image_as_patches(c_path)

        # stack patches into a single tensor
        patches = torch.stack([
            patches_a, patches_b, patches_c
        ], dim=1)

        # normalise patches
        patches = patches / 255.0

        if self.norm:
            patches = self.normalise(patches)

        if self.transform is not None:
            patches = self.transform(patches)

        return patches



    

