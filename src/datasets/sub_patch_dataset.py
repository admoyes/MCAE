from random import shuffle
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from skimage.util import view_as_windows
from .utils.patches import get_patches
from collections import defaultdict
torch.manual_seed(0)

class TripletPatchDataset(Dataset):

    def __init__(self, root_path: str, patch_size: int, norm: bool, transform=None, shuffle_patches=True) -> None:
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
        self.shuffle_patches = shuffle_patches

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
        im = np.array(Image.open(path).convert("RGB")).astype(float)
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
        ], dim=1).float()

        # normalise patches
        patches = patches / 255.0

        # shuffle patches
        patch_idx = torch.randperm(patches.shape[0])
        patches = patches[patch_idx]

        if self.norm:
            patches = self.normalise(patches)

        if self.transform is not None:
            patches = self.transform(patches)

        return patches



    
class BasicPatchDataset(Dataset):

    def __init__(self,
            image_folder: str,
            patch_size: int,
            image_file_extension: str,
            patch_transform=None) -> None:

        self.image_folder = Path(image_folder)
        self.patch_size = patch_size
        self.patch_transform = patch_transform
        
        # find all the classes
        class_folders = list(self.image_folder.iterdir())

        # find images belonging to each class
        class_paths = defaultdict(list)
        for class_folder in class_folders:
            class_name = class_folder.name
            class_image_paths = list(class_folder.iterdir())
            class_paths[class_name] = class_image_paths

        # build a list of examples and their classes
        self.class_names = list(class_paths.keys())
        self.examples = []
        for class_name, class_path_list in class_paths.items():

            # create a list of <image_path, class_index> pairs.
            class_index = self.class_names.index(class_name)
            class_indices = [class_index] * len(class_path_list)
            class_examples = list(zip(class_path_list, class_indices))
            
            # save to examples list
            self.examples.extend(class_examples)

    def process_image(self, image_path: Path) -> torch.FloatTensor:
        """
        Convert the image at the given path into a feature vector. 
        """

        # load the image
        im = np.array(Image.open(image_path).convert("RGB")).astype(float)
        im = im / 255.0

        # extract patches
        patches = get_patches(im, self.patch_size)

        return patches


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        image_path, class_index = self.examples[index]
        patches = self.process_image(image_path)

        if self.patch_transform is not None:
            patches = self.patch_transform(patches)

        return patches
