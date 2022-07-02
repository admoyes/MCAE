import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image
from .utils.patches import get_patches


class ClassificationDataset(Dataset):

    def __init__(self, 
            image_folder: str,
            patch_size: int,
            image_file_extension: str,
            autoencoder: nn.Module) -> None:
        """
        Dataset used for image classification. 

        This dataset will read in a series of large images (e.g. 768x768).
        The MCAE/StaNoSA models are most likely trained on patches sized 8x8.
        Therefore, this dataset will process the images as follows:
        1. Extract all PxP (where P = `patch_size`) patches.
        2. Use given encoder model to extract a feature vector for each PxP patch.
        3. Average the feature vectors for each PxP patch to produce an image-level \
           feature vector.

        Parameters
        ----------
        image_folder: str
            Path to the root directory of the dataset. This folder should contain only folders named after each class,\
                e.g. /dataset_root_dir/class_a, /dataset_root_dir/class_b, etc \
                and each of these sub-folders should contain only images of the extension specified by \
                the `image_folder_extension` arguement.
        patch_size: int
            The size of patch to extract from each image. Should be the same as was used to train the feature \
                extraction model (e.g. MCAE or StaNoSA).
        image_file_extension: str
            The extension that each of the images should have, e.g. 'jpeg'.
        autoencoder: nn.Module
            The PyTorch model to use to extract features from each patch. i.e. MCAE or StaNoSA.
        """

        self.image_folder = Path(image_folder)
        self.patch_size = patch_size
        self.autoencoder = autoencoder
        
        # find all the classes
        class_folders = list(self.image_folder.iterdir())

        # find images belonging to each class
        class_paths = defaultdict[list]
        for class_folder in class_folders:
            class_name = class_folder.name
            class_image_paths = list(class_folder.glob(f"*.{image_file_extension}"))
            class_paths[class_name] = class_image_paths

    
    def process_image(self, image_path: Path) -> torch.FloatTensor:
        """"
        Convert the image at the given path into a feature vector.
        """

        # load the image
        im = np.array(Image.open(image_path))

        # extract patches
        patches = get_patches(im, self.patch_size)

        # convert patches into features
        with torch.no_grad():
            features = self.autoencoder.encode(patches)
         
        # average patch features into image-level feature 
        image_feature = features.mean(dim=0)

        # return feature vector
        return image_feature

