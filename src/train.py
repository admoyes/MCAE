import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from datasets import TripletPatchDataset
from models.mcae import train_mcae
from models.stanosa import train_stanosa
import mlflow
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model-type", type=str, required=True, help="mcae or stanosa or dcae")
parser.add_argument("--epochs", type=int, default=100, help="num training epochs")
parser.add_argument("--batch-size", type=int, default=32, help="batch size")
parser.add_argument("--patch-size", type=int, default=8, help="patch size")
parser.add_argument("--num-workers", type=int, default=6, help="num workers for data loader")
parser.add_argument("--stanosa-domain-index", type=int, default=-1, help="which domain of the triplet dataset to use for training stanosa")
parser.add_argument("--dcae-domain-index-1", type=int, default=-1, help="first DCAE domain")
parser.add_argument("--dcae-domain-index-2", type=int, default=-1, help="second DCAE domain")
args = parser.parse_args()


if args.model_type == "stanosa":

    # check the domain index is given
    if args.stanosa_domain_index < 0 or args.stanosa_domain_index > 2:
        raise ValueError(f"stanosa domain index should be 0, 1, or 2. got {args.stanosa_domain_index}")

    # define stanosa dataset and dataloader
    dataset = TripletPatchDataset("/data/triplet_dataset/", args.patch_size, False)

    dataloader = DataLoader(
        dataset,
        num_workers=args.num_workers,
        shuffle=True,
        batch_size=args.batch_size,
    )

    # train model
    train_stanosa(dataloader, args.stanosa_domain_index, epochs=args.epochs)

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

    # check the domain indices are valid
    if args.dcae_domain_index_1 < 0 or args.dcae_domain_index_1 > 2:
        raise ValueError(f"dcae domain index 1 should be 0, 1, or 2. got {args.dcae_domain_index_1}")

    if args.dcae_domain_index_2 < 0 or args.dcae_domain_index_2 > 2:
        raise ValueError(f"dcae domain index 2 should be 0, 1, or 2. got {args.dcae_domain_index_2}")

    if args.dcae_domain_index_1 == args.dcae_domain_index_2:
        raise ValueError("dcae domain indices should not be equal")

    # define transform that selects on the correct domains
    domain_index = torch.LongTensor([
        args.dcae_domain_index_1,
        args.dcae_domain_index_2
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