# Testing our customized ResNet18flex_dual on melpcen and mfcc

# Features: mel pcen spectrograms and MFCCs
# Architecture: ResNet18 (modified for our inputs)
# Inputs: Spectrograms of dimension 256x128 and 256x32
# Model: Best model from training
# Loss function: Soft Labeling Loss
# Target: ~250 musicmap genres

###===Initial Setup===

# Imports
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import one_hot
from torch import nn, optim
from torch.optim.swa_utils import AveragedModel
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
from bs4 import BeautifulSoup
from collections import defaultdict
from pyvis.network import Network
from tabulate import tabulate
from pathlib import Path
import contextlib
import argparse
import sys
from torch.cuda.amp import autocast

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
	"--dir",
	type=str,
	dest="dir",
)
args = parser.parse_args()

# Add the directory to sys.path to allow local imports
sys.path.append(args.dir)

# Change working directory so later relative paths work
os.chdir(args.dir)


# Local Imports
from musicmap_graph_creator import create_musicmap_graph, compute_shortest_paths_between_classes_nx
from model_resnet18flex_dual import DualBranchResNet18Gray
from musicmap_helpers import compute_image_dataset_stats, check_set_gpu, batch_soft_labeling_loss, top_k_accuracy, evaluate_dual, plot_metrics, plot_cm_super, get_input_height, PairedFeatureDataset

### Global Parameters
# Set the device to the local GPU if available
device = check_set_gpu()

num_epochs = 100	# Maximum number of epochs to try
beta = 1.5			# Hyperparameter for soft loss function
patience = 10		# Number of epochs in early stopping condition
swa_start_epoch = 5	 # start averaging after this epoch

# Edge weights for the graph
primary_edge_weight = 1
secondary_edge_weight = 2
backlash_edge_weight = 3
supergenre_edge_weight = 4
world_supergenre_weight = 1
world_cluster_weight = 1
util_genre_weight = 2
cluster_weight = 1

###===Compute the Shortest Path Matrix===
# This uses the create_musicmap_graph and compute_shortest_paths_between_classes_nx functions from the musicmap_graph_creator file

musicmap_graph = create_musicmap_graph(
	primary_edge_weight, 
	secondary_edge_weight,
	backlash_edge_weight,
	supergenre_edge_weight,
	world_supergenre_weight,
	world_cluster_weight,
	util_genre_weight,
	cluster_weight
)

shortest_graph, class_names = compute_shortest_paths_between_classes_nx(
	class_dir="./15_second_features_augmented/musicmap_processed_output_splits_train/mel_pcen_gray",  # path to folder with class folders
	graph=musicmap_graph,
	return_tensor=True
)

# Base directories for test
test_dir1 = "./15_second_features_augmented/musicmap_processed_output_splits_test/mel_pcen_gray"
test_dir2 = "./15_second_features_augmented/musicmap_processed_output_splits_test/mfcc_gray"


output_dir = Path("testing_ResNet_outputs_pcen_mfcc")
output_dir.mkdir(parents=True, exist_ok=True)

###===Transform===
# We'll use the means and standard deviations from the training data to normalize the images.
# Using ImageFolder might default to loading images as RGB, even if they're just greyscale. 
# So we'll force greyscale as part of the transform for greyscale images.

mel_mean = torch.tensor([0.09798076748847961])
mel_std = torch.tensor([0.10669851303100586])
mfcc_mean = torch.tensor([0.6104416847229004])
mfcc_std = torch.tensor([0.13878336548805237])

transform1 = transforms.Compose([
	transforms.ToTensor(),
	transforms.Grayscale(num_output_channels=1),
	transforms.Normalize(mean=mel_mean, std=mel_std),
])
transform2 = transforms.Compose([
	transforms.ToTensor(),
	transforms.Grayscale(num_output_channels=1),
	transforms.Normalize(mean=mfcc_mean, std=mfcc_std),
])


###===Load the Data===
# Define the datasets and create dataloaders.
# ImageFolder automatically creates labels from the folder structure

test_dataset = PairedFeatureDataset(test_dir1, test_dir2, transform1, transform2)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, prefetch_factor=2)


###===Architecture===
# Compute the number of classes, compute the height of the spectrogram, and instantiate the model

num_classes = len(test_dataset.classes)

height1 = get_input_height(test_dir1)
height2 = get_input_height(test_dir2)

model_local = DualBranchResNet18Gray(input_height1=height1, input_height2=height2, num_classes=num_classes).to(device)
model_local = model_local.to(device)
model_local.load_state_dict(torch.load("best_model.pth"))

###===Criterion ===
criterion = lambda outputs, labels: batch_soft_labeling_loss(outputs, labels, shortest_graph, beta)
	

###===Testing Loop===
# This calls an evaluate function, which returns a dictionary of metrics
# Incorporates auto-stopping based on mean distance
# Automatically saves a best model, a SWA model, a plot of evaluation metrics, and a plot of a confusion matrix.

model_local.eval()

idx_to_supergenre = np.array([cls[:3] for cls in test_dataset.classes])	 # shape (num_classes,)

test_metrics = evaluate_dual(
	idx_to_supergenre=idx_to_supergenre,
	model=model_local,
	dataloader=test_dataloader,
	criterion=criterion,
	device=device,
	shortest_graph=shortest_graph,
	train_dataset=test_dataset
)

# Print a metrics summary
headers = ["Test Acc", "Top-3 Acc", "Top-5 Acc", "Precision", "Recall", "F1", "Supergenre Acc", "Top-3 Super", "Mean Dist", "Std Dist"]
row = [
	f"{test_metrics['val_accuracy']:.2%}",
	f"{test_metrics['top3_accuracy']:.2%}",
	f"{test_metrics['top5_accuracy']:.2%}",
	f"{test_metrics['precision']:.2%}",
	f"{test_metrics['recall']:.2%}",
	f"{test_metrics['f1']:.2%}",
	f"{test_metrics['supergenre_accuracy']:.2%}",
	f"{test_metrics['supergenre_top3_accuracy']:.2%}",
	f"{test_metrics['mean_distance']:.4f}",
	f"{test_metrics['std_distance']:.4f}",
]
print(tabulate([row], headers=headers, tablefmt="grid"))

# Save the metrics in a df
metrics_df = pd.DataFrame({
	"val_accuracy": test_metrics['val_accuracy'],
	"top3_accuracy": test_metrics['top3_accuracy'],
	"top5_accuracy": test_metrics['top5_accuracy'],
	"precision": test_metrics['precision'],
	"recall": test_metrics['recall'],
	"f1": test_metrics['f1'],
	"supergenre_accuracy": test_metrics['supergenre_accuracy'],
	"supergenre_top3_accuracy": test_metrics['supergenre_top3_accuracy'],
	"mean_distance": test_metrics['mean_distance'],
	"std_distance": test_metrics['std_distance']
})
metrics_df.to_csv(output_dir / "testing_metrics.csv", index=False)

# Save confusion matrices
cm_genre = test_metrics['genre_confusion_matrix']
cm_genre.to_csv(output_dir / "confusion_matrix_genre_TEST.csv", index=True)

cm_supergenre = test_metrics['supergenre_confusion_matrix']
cm_supergenre.to_csv(output_dir / "confusion_matrix_supergenre_TEST.csv", index=True)

# Percent versions
cm_genre_pct = cm_genre.div(cm_genre.sum(axis=1), axis=0) * 100
cm_genre_pct.to_csv(output_dir / "confusion_matrix_genre_TEST_percent.csv", index=True)

cm_supergenre_pct = cm_supergenre.div(cm_supergenre.sum(axis=1), axis=0) * 100
cm_supergenre_pct.to_csv(output_dir / "confusion_matrix_supergenre_TEST_percent.csv", index=True)

# Heatmaps
plot_cm_super(cm_supergenre, output_dir/"supergenre_confusion_matrix_TEST.png")
plot_cm_super(cm_supergenre_pct, output_dir/"supergenre_confusion_matrix_TEST_percent.png", percent=True)
