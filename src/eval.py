import torch
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from datasets import ClassificationDataset, BasicPatchDataset
from datasets.utils.sample import sample_from_dataloader
from datasets.utils.ZCA import compute_zca_matrix

from models.stanosa import eval_stanosa
from models.mcae import eval_mcae
from umap import UMAP
from matplotlib import pyplot as plt
from tqdm import tqdm
from einops import rearrange
import mlflow
import argparse
from mlflow.tracking.client import MlflowClient

parser = argparse.ArgumentParser()
parser.add_argument("--model-type", type=str, required=True, help="mcae or stanosa or dcae")
parser.add_argument("--batch-size", type=int, default=32, help="batch size")
parser.add_argument("--patch-size", type=int, default=8, help="patch size")
parser.add_argument("--classifier", type=str, required=True, help="mlp or rf or svm")
parser.add_argument("--training-experiment-name", type=str, required=True, help="training experiment name")
parser.add_argument("--training-run-name", type=str, required=True, help="training run name")
parser.add_argument("--domain-index", type=str, default=-1, help="which autoencoder to use (for MCAE and DCAE)")
parser.add_argument("--dataset-path", type=str, required=True, help="kather or lung or colon")
args = parser.parse_args()

print("EVAL!")
print(args, flush=True)

if args.model_type == "stanosa":

    eval_stanosa(
        args.dataset_path,
        args.classifier,
        args.training_experiment_name,
        args.training_run_name,
        args.batch_size,
        args.patch_size
    ) 

elif args.model_type in ["mcae", "dcae"]:

    eval_mcae(
        args.dataset_path,
        args.classifier,
        args.training_experiment_name,
        args.training_run_name,
        args.batch_size,
        args.patch_size,
        int(args.domain_index)
    )

else:
    raise ValueError(f"unrecognised model type. got '{args.model_type}'")
#mcae = MultiChannelAutoEncoderLinear(3)
#
#lung_dataset = ClassificationDataset(
#    "/data/lung_images/",
#    8, "jpeg", mcae,
#    None,
#    model_call_args={
#        "index": torch.zeros(1).fill_(0) # directs calls to the 0th auto encoder in the MCAE
#    }
#)
#
#dataloader = DataLoader(lung_dataset, batch_size=64, num_workers=8)
#
#all_features = []
#all_labels = []
#for (feature, label) in tqdm(dataloader, total=len(dataloader)):
#    all_features.append(feature)
#    all_labels.append(label)
#
#all_features = torch.cat(all_features, dim=0)
#all_labels = torch.cat(all_labels)
#
#reducer = UMAP(n_components=2)
#reducer.fit(all_features)
#embedded = reducer.embedding_
#fig, ax = plt.subplots(figsize=(10,10))
#ax.scatter(
#    embedded[:, 0],
#    embedded[:, 1],
#    c=all_labels,
#    s=3
#)
#plt.savefig("/output/umap_plot.jpeg")