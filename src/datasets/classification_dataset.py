import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as tf
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image
from .utils.patches import get_patches
from typing import Callable
from einops import rearrange, reduce
from collections import Counter


class ClassificationDataset(Dataset):

    def __init__(self, 
            image_folder: str,
            patch_size: int,
            image_file_extension: str,
            autoencoder: nn.Module,
            patch_transform: Callable[[torch.FloatTensor], torch.FloatTensor],
            model_call_args: dict = {}) -> None:
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
        self.patch_transform = patch_transform
        self.model_call_args = model_call_args
        
        # find all the classes
        class_folders = list(self.image_folder.iterdir())

        # find images belonging to each class
        class_paths = defaultdict(list)
        for class_folder in class_folders:
            class_name = class_folder.name
            class_image_paths = list(class_folder.iterdir())
            class_paths[class_name] = class_image_paths

        # build a list of examples and their classes
        self.class_names = list(sorted(class_paths.keys()))
        self.examples = []
        for class_name, class_path_list in class_paths.items():

            # create a list of <image_path, class_index> pairs.
            class_index = self.class_names.index(class_name)
            class_indices = [class_index] * len(class_path_list)
            class_examples = zip(class_path_list, class_indices)
            
            # save to examples list
            self.examples.extend(class_examples)

        # count labels 
        class_counts = Counter([ex[1] for ex in self.examples])
        print(class_counts, flush=True)

    
    def process_image(self, image_path: Path) -> torch.FloatTensor:
        """"
        Convert the image at the given path into a feature vector.
        """

        # load the image
        im = np.array(Image.open(image_path).convert("RGB")).astype(float)
        im = im / 255.0
        #print("im", im.shape)

        # extract patches
        patches = get_patches(im, self.patch_size)
        #print("patches", patches.shape)

        # apply a transform
        # e.g. global contrast normalisation for StaNoSA
        if self.patch_transform is not None:
            patches = self.patch_transform(patches)

        # insert batch and domain dimensions so it can be passed through the model
        patches = rearrange(patches, "p f w h -> 1 p f w h")
        #print("patches", patches.shape)

        # convert patches into features
        with torch.no_grad():
            features, _ = self.autoencoder(patches, **self.model_call_args)
            #print("features", features.shape)
            # reduce unnecessary dimensions
            features = features.squeeze()
         
        # average patch features into image-level feature 
        image_feature = reduce(features, "b f -> f", "mean")

        # return feature vector
        return image_feature


    def __len__(self):
        return len(self.examples)


    def __getitem__(self, index):
        image_path, class_index = self.examples[index]
        return self.process_image(image_path), class_index
