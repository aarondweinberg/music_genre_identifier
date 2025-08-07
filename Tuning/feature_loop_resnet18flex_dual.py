# Training ResNet18 with a variety of pairs of features

# Features: This loop works with pairs of features
# 			mel spectrograms, chroma, chroma_bs, chroma_cq, hpss_mean, hpss_median, and mfcc
#           from 15-second clips, each augmented with pitch shift and pink noise
# Architecture: ResNet18 (modified for our inputs)
# Inputs: A variety of spectrograms of dimension 12x256, 32x256, and 128x256 (also graphs of dimension 300x300)
#         We'll adjust the ResNet18 architecture use stride=1 in conv1 and skip maxpool when the height is 12 or 32
# Model: Tuning from scratch
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
#from torch.amp import GradScaler
from torch.cuda.amp import GradScaler	# This version is deprecated, but the other is throwing errors...
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

from itertools import combinations


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

# Global Parameters
# Set the device to the local GPU if available
device = check_set_gpu()

num_epochs = 20  # Maximum number of epochs to try
beta = 2        # Hyperparameter for soft loss function
patience = 5   # Number of epochs in early stopping condition
swa_start_epoch = 5  # start averaging after this epoch

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
	class_dir="./15_second_features_augmented/musicmap_processed_output_splits_train/chroma_bs_gray",  # path to folder with class folders
	graph=musicmap_graph,
	return_tensor=True
)

# Base directories for train and val
train_base = "./15_second_features_augmented/musicmap_processed_output_splits_train"
val_base = "./15_second_features_augmented/musicmap_processed_output_splits_val"

# List of directories to exclude
exclude_features = {"features_csv", "resnet_mel_rgb"}

# Get valid feature sets
all_feature_dirs = sorted([
	d for d in os.listdir(train_base)
	if os.path.isdir(os.path.join(train_base, d)) and d not in exclude_features
])

# Create all 2-feature combinations
#feature_pairs = list(combinations(all_feature_dirs, 2))

# Create all 2-feature combinations with mel_pcen_gray
feature_pairs = [
    ('mel_pcen_gray', f2) for f2 in all_feature_dirs
    if f2 != 'mel_pcen_gray' and f2 not in exclude_features
]

# Loop over subfolders inside train_base (assuming val_base has matching structure)
for f1, f2 in feature_pairs:
	if f1 in exclude_features or f2 in exclude_features:
		print(f"Skipping pair with excluded feature: {f1} + {f2}")
		continue
	print(f"=== Training on pair: {f1} + {f2} ===")

	train_dir1 = os.path.join(train_base, f1)
	train_dir2 = os.path.join(train_base, f2)
	val_dir1 = os.path.join(val_base, f1)
	val_dir2 = os.path.join(val_base, f2)

	height1 = get_input_height(train_dir1)
	height2 = get_input_height(train_dir2)

	dataset_name = f"{f1}__{f2}"
	output_dir = Path("dual_feature_outputs") / dataset_name
	output_dir.mkdir(parents=True, exist_ok=True)


	###===Transform===
	# We'll use the means and standard deviations to normalize the images.
	# Using ImageFolder might default to loading images as RGB, even if they're just greyscale. 
	# So we'll force greyscale as part of the transform for greyscale images.
	
	means1, stds1 = compute_image_dataset_stats(train_dir1)
	means2, stds2 = compute_image_dataset_stats(train_dir2)
	
	transform1 = transforms.Compose([
		transforms.ToTensor(),
		transforms.Grayscale(num_output_channels=1),
		transforms.Normalize(mean=means1, std=stds1),
	])
	transform2 = transforms.Compose([
		transforms.ToTensor(),
		transforms.Grayscale(num_output_channels=1),
		transforms.Normalize(mean=means2, std=stds2),
	])
	
	
	###===Load the Data===
	# Define the datasets and create dataloaders.
	# ImageFolder automatically creates labels from the folder structure
	
	train_dataset = PairedFeatureDataset(train_dir1, train_dir2, transform1, transform2)
	val_dataset = PairedFeatureDataset(val_dir1, val_dir2, transform1, transform2)

	train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
	val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)


	###===Architecture===
	# Compute the number of classes and instantiate the model
	# In this case, we're importing the ResNet18Gray model from an external file
	
	num_classes = len(train_dataset.classes)
	model_local = DualBranchResNet18Gray(input_height1=height1, input_height2=height2, num_classes=num_classes).to(device)
	
	###===Criterion, Optimizer, and Scheduler===
	criterion = lambda outputs, labels: batch_soft_labeling_loss(outputs, labels, shortest_graph, beta)
	
	optimizer = optim.Adam(model_local.parameters(), lr=0.001, weight_decay=1e-4)	
	
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, mode='max', factor=0.5, patience=2)#, verbose=True)
		
	
	###===Training Loop===
	# This calls an evaluate function, which automatically saves a confusion matrix.
	# Incorporates auto-stopping based on mean distance
	# Automatically saves a best model, a SWA model, a plot of evaluation metrics, and a plot of a confusion matrix.
	
	# Create index-to-supergenre lookup once
	idx_to_supergenre = np.array([cls[:3] for cls in train_dataset.classes])  # shape (num_classes,)
	
	# Metric lists
	val_accuracy_list, top3_acc_list, top5_acc_list = [], [], []
	precision_list, recall_list, f1_list = [], [], []
	mean_dist_list, super_acc_list, super_top3_acc_list = [], [], []
	train_loss_list, train_accuracy_list, val_loss_list = [], [], []
	
	# Early stopping and SWA setup
	#best_val_accuracy = 0.0
	# Going to stop based on mean distance rather than val accuracy
	best_mean_distance = float('inf')
	best_model_state = None
	counter = 0
	swa_model = AveragedModel(model_local)  # model to store average weights
	
	use_amp = torch.cuda.is_available()
	
	log_path = output_dir / "training_log.txt"
	with open(log_path, "w") as log_file:
		with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
		
			
			# === Training Loop ===
			for epoch in range(num_epochs):
				model_local.train()
				train_loss, correct, total = 0.0, 0, 0

				scaler = GradScaler(enabled=torch.cuda.is_available())	# This version is deprecated, but the other is throwing errors...
				#scaler = GradScaler(
				#	enabled=torch.cuda.is_available(),
				#	device_type='cuda' if torch.cuda.is_available() else 'cpu'
				#)

				for batch_idx, ((input1, input2), labels) in enumerate(train_dataloader):  # <-- unpack tuple
					input1, input2, labels = input1.to(device), input2.to(device), labels.to(device)
					optimizer.zero_grad()

					if use_amp:
						with torch.amp.autocast('cuda'):
							outputs = model_local((input1, input2))  # <-- dual input
							outputs = torch.log_softmax(outputs, dim=1)
							loss = criterion(outputs, labels)
					else:
						outputs = model_local((input1, input2))  # <-- dual input
						outputs = torch.log_softmax(outputs, dim=1)
						loss = criterion(outputs, labels)

					if use_amp:					
						scaler.scale(loss).backward()
						scaler.step(optimizer)
						scaler.update()
					else:
						loss.backward()
						optimizer.step()

					train_loss += loss.item()
					_, predicted = outputs.max(1)
					correct += predicted.eq(labels).sum().item()
					total += labels.size(0)

				avg_train_loss = train_loss / len(train_dataloader)
				train_accuracy = correct / total
				train_loss_list.append(avg_train_loss)
				train_accuracy_list.append(train_accuracy)

				# --- Validation ---
				metrics = evaluate_dual(idx_to_supergenre, model_local, val_dataloader, criterion, device, shortest_graph, train_dataset)

				# Early stopping
				if metrics['mean_distance'] < best_mean_distance:
					best_mean_distance = metrics['mean_distance']
					best_model_state = model_local.state_dict()
					counter = 0
					print(f"[Epoch {epoch+1}] New best mean_distance: {best_mean_distance:.4f}")
				else:
					counter += 1
					print(f"No improvement in mean_distance for {counter} epoch(s).")

				# SWA
				if epoch >= swa_start_epoch:
					swa_model.update_parameters(model_local)

				# Append metrics
				for lst, key in zip(
					[val_loss_list, val_accuracy_list, top3_acc_list, top5_acc_list, precision_list, recall_list, f1_list,
					mean_dist_list, super_acc_list, super_top3_acc_list],
					["val_loss", "val_accuracy", "top3_accuracy", "top5_accuracy", "precision", "recall", "f1",
					"mean_distance", "supergenre_accuracy", "supergenre_top3_accuracy"]
				):
					lst.append(metrics[key])

				# Print metrics table
				headers = ["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc",
						"Top-3 Acc", "Top-5 Acc", "Precision", "Recall", "F1",
						"Supergenre Acc", "Top-3 Super", "Mean Dist"]

				row = [epoch + 1,
					f"{avg_train_loss:.4f}", f"{train_accuracy:.2%}",
					f"{metrics['val_loss']:.4f}", f"{metrics['val_accuracy']:.2%}",
					f"{metrics['top3_accuracy']:.2%}", f"{metrics['top5_accuracy']:.2%}",
					f"{metrics['precision']:.2%}", f"{metrics['recall']:.2%}", f"{metrics['f1']:.2%}",
					f"{metrics['supergenre_accuracy']:.2%}", f"{metrics['supergenre_top3_accuracy']:.2%}",
					f"{metrics['mean_distance']:.4f}"]

				try:
					print(tabulate([row], headers=headers, tablefmt="grid"))
				except Exception as e:
					print(f"Error printing metrics: {e}", flush=True)
					print(f"metrics dict: {metrics}")
					raise

				if counter >= patience:
					print(f"Early stopping triggered after {patience} epochs of no improvement.")
					break

				scheduler.step(metrics['mean_distance'])

			# Save best model
#			if best_model_state is not None:
#				torch.save(best_model_state, output_dir / 'best_model.pth')
#				print(f"Best model saved with mean_distance: {best_mean_distance:.4f}")

			# Save SWA model
#			torch.save(swa_model.module.state_dict(), output_dir / 'swa_model.pth')
#			print("SWA model saved from averaged checkpoints.")

			# Save training metrics
			metrics_df = pd.DataFrame({
				"epoch": list(range(1, len(train_loss_list) + 1)),
				"train_loss": train_loss_list,
				"train_accuracy": train_accuracy_list,
				"val_loss": val_loss_list,
				"val_accuracy": val_accuracy_list,
				"top3_accuracy": top3_acc_list,
				"top5_accuracy": top5_acc_list,
				"precision": precision_list,
				"recall": recall_list,
				"f1": f1_list,
				"supergenre_accuracy": super_acc_list,
				"supergenre_top3_accuracy": super_top3_acc_list,
				"mean_distance": mean_dist_list
			})
			metrics_df.to_csv(output_dir / "training_metrics.csv", index=False)

			# Plot metrics
			plot_metrics(val_accuracy_list, top3_acc_list, top5_acc_list, precision_list, recall_list, f1_list, mean_dist_list, super_acc_list, super_top3_acc_list, output_dir / "metrics.png")

			# Save confusion matrices
			cm_genre = metrics['genre_confusion_matrix']
			cm_genre.to_csv(output_dir / "confusion_matrix_genre.csv", index=True)

			cm_supergenre = metrics['supergenre_confusion_matrix']
			cm_supergenre.to_csv(output_dir / "confusion_matrix_supergenre.csv", index=True)

			# Plot supergenre confusion matrix
			plot_cm_super(cm_supergenre, output_dir / "supergenre_confusion_matrix.png")