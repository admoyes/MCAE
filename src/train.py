import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from datasets import TripletPatchDataset
from models.mcae import train_mcae
from models.stanosa import train_stanosa
from datasets.utils.norm import global_contrast_normalisation
import mlflow
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model-type", type=str, required=True, help="mcae or stanosa or dcae")
parser.add_argument("--epochs", type=int, default=100, help="num training epochs")
parser.add_argument("--batch-size", type=int, default=256, help="batch size")
parser.add_argument("--patch-size", type=int, default=8, help="patch size")
parser.add_argument("--num-workers", type=int, default=6, help="num workers for data loader")
parser.add_argument("--domain-index-1", type=str, default=-1, help="first DCAE/StaNoSA domain")
parser.add_argument("--domain-index-2", type=str, default=-1, help="second DCAE domain")
args = parser.parse_args()


if args.model_type == "stanosa":

    stanosa_domain_index = int(args.domain_index_1)
    # check the domain index is given
    if stanosa_domain_index < 0 or stanosa_domain_index > 2:
        raise ValueError(f"stanosa domain index should be 0, 1, or 2. got {stanosa_domain_index}")

    # define stanosa dataset and dataloader
    dataset = TripletPatchDataset(
        "/data/triplet_dataset/",
        args.patch_size,
        False,
        transform=tf.Lambda(lambda patch: global_contrast_normalisation(patch))
    )

    dataloader = DataLoader(
        dataset,
        num_workers=args.num_workers,
        shuffle=True,
        batch_size=args.batch_size,
    )

    # train model
    train_stanosa(dataloader, stanosa_domain_index, epochs=args.epochs)

elif args.model_type == "mcae":

    # define the MCAE dataset and dataloader
    dataset = TripletPatchDataset("/data/triplet_dataset/", args.patch_size, True)

    dataloader = DataLoader(
        dataset,
        num_workers=args.num_workers,
        shuffle=True,
        batch_size=args.batch_size,
    )

    # train MCAE model
    train_mcae(dataloader, epochs=args.epochs)

elif args.model_type == "dcae":
    dcae_domain_index_1 = int(args.domain_index_1)
    dcae_domain_index_2 = int(args.domain_index_2)

    # check the domain indices are valid
    if dcae_domain_index_1 < 0 or dcae_domain_index_1 > 2:
        raise ValueError(f"dcae domain index 1 should be 0, 1, or 2. got {dcae_domain_index_1}")

    if dcae_domain_index_2 < 0 or dcae_domain_index_2 > 2:
        raise ValueError(f"dcae domain index 2 should be 0, 1, or 2. got {dcae_domain_index_2}")

    if dcae_domain_index_1 == dcae_domain_index_2:
        raise ValueError("dcae domain indices should not be equal")

    # define transform that selects on the correct domains
    domain_index = torch.LongTensor([
        dcae_domain_index_1,
        dcae_domain_index_2
    ])
    transform = tf.Lambda(lambda patches: torch.index_select(patches, 1, domain_index))

    # define the DCAE dataset and dataloader
    dataset = TripletPatchDataset(
        "/data/triplet_dataset/",
        args.patch_size,
        True,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        num_workers=args.num_workers,
        shuffle=True,
        batch_size=args.batch_size,
    )

    # train DCAE model
    train_mcae(dataloader, num_domains=2, epochs=args.epochs)
else:
    raise ValueError(f"model type must be 'mcae' or 'stanosa'. got {args.model_type}")