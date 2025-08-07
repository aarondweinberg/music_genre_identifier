# Training our customized ResNet18

# Features: mel pcen spectrograms
# Architecture: ResNet18 (modified for our inputs)
# Inputs: Mel PCEN Spectrograms of dimension 256x128
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
from model_resnet18flex import ResNet18Gray
from musicmap_helpers import compute_image_dataset_stats, check_set_gpu, batch_soft_labeling_loss, top_k_accuracy, evaluate, plot_metrics, plot_cm_super, get_input_height

### Global Parameters
# Set the device to the local GPU if available
device = check_set_gpu()

num_epochs = 100  	# Maximum number of epochs to try
beta = 2.5        	# Hyperparameter for soft loss function
patience = 10   	# Number of epochs in early stopping condition
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
	class_dir="./15_second_features_augmented/musicmap_processed_output_splits_train/chroma_cq_gray",  # path to folder with class folders
	graph=musicmap_graph,
	return_tensor=True
)

# Base directories for train and val
train_data_directory = "./15_second_features_augmented/musicmap_processed_output_splits_train/mel_pcen_gray"
val_data_directory = "./15_second_features_augmented/musicmap_processed_output_splits_val/mel_pcen_gray"

output_dir = Path("trained_ResNet_outputs") 
output_dir.mkdir(parents=True, exist_ok=True)


###===Transform===
# We'll use the means and standard deviations to normalize the images.
# Using ImageFolder might default to loading images as RGB, even if they're just greyscale. 
# So we'll force greyscale as part of the transform for greyscale images.

image_dir = train_data_directory
means, sds = compute_image_dataset_stats(image_dir)

transform_greyscale = transforms.Compose([
	transforms.ToTensor(), # Should always be the last step before feeding into a model
	transforms.Grayscale(num_output_channels=1),  # force grayscale
	transforms.Normalize(mean=means, std=sds)    # Normalize to imagenet mean and standard deviation
])


###===Load the Data===
# Define the datasets and create dataloaders.
# ImageFolder automatically creates labels from the folder structure

train_dataset = torchvision.datasets.ImageFolder(
	root=train_data_directory,
	transform=transform_greyscale
)

val_dataset = torchvision.datasets.ImageFolder(
	root=val_data_directory,
	transform=transform_greyscale
)

train_dataloader = torch.utils.data.DataLoader(
	train_dataset,
	batch_size=64,  # Adjust batch size as needed
	shuffle=True,
	num_workers=4,  # Adjust this to tweak multiprocessing
	pin_memory=True,
	prefetch_factor=2
)

val_dataloader = torch.utils.data.DataLoader(
	val_dataset,
	batch_size=10,  # Adjust batch size as needed
	shuffle=False, #Setting to false keeps evaluation stable across epochs
	num_workers=4,
	pin_memory=True,
	prefetch_factor=2 
)


###===Architecture===
# Compute the number of classes, compute the height of the spectrogram, and instantiate the model

num_classes = len(train_dataloader.dataset.classes) 

input_height = get_input_height(train_data_directory)

model_local = ResNet18Gray(num_classes=num_classes, input_height=input_height)
model_local = model_local.to(device)


###===Criterion, Optimizer, and Scheduler===
criterion = lambda outputs, labels: batch_soft_labeling_loss(outputs, labels, shortest_graph, beta)

optimizer = optim.Adam(model_local.parameters(), lr=0.001, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
	optimizer, mode='max', factor=0.5, patience=2)#, verbose=True)
	

###===Training Loop===
# This calls an evaluate function, which returns a dictionary of metrics
# Incorporates auto-stopping based on mean distance
# Automatically saves a best model, a SWA model, a plot of evaluation metrics, and a plot of a confusion matrix.

# Create index-to-supergenre lookup once
idx_to_supergenre = np.array([cls[:3] for cls in train_dataset.classes])  # shape (num_classes,)

# Metric lists
val_accuracy_list, top3_acc_list, top5_acc_list = [], [], []
precision_list, recall_list, f1_list = [], [], []
mean_dist_list, std_dist_list, super_acc_list, super_top3_acc_list = [], [], [], []
train_loss_list, train_accuracy_list, val_loss_list = [], [], []

# Early stopping and SWA setup
# Going to stop based on mean distance rather than val accuracy
best_mean_distance = float('inf')
best_model_state = None
counter = 0
swa_model = AveragedModel(model_local)  # model to store average weights

# Save all output to a log file for debugging
log_path = output_dir / "training_log.txt"
with open(log_path, "w") as log_file:
	with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
	
		# === Training Loop ===
		for epoch in range(num_epochs):
			model_local.train()
			train_loss, correct, total = 0.0, 0, 0
		
			# GradScaler might speed up cuda, but is not available on mps
			scaler = GradScaler(enabled=torch.cuda.is_available())
		
			# Train
			for batch_idx, (inputs, labels) in enumerate(train_dataloader):
				inputs, labels = inputs.to(device), labels.to(device)
				optimizer.zero_grad()
		
				# Putting this inside autocast might speed up cuda computations
				if torch.cuda.is_available():
					with torch.amp.autocast('cuda'):
						outputs = model_local(inputs)
		
						# Take the log softmax of outputs to use as inputs to the loss function
						outputs = torch.log_softmax(outputs, dim=1)
						loss = criterion(outputs, labels)
				else:
					outputs = model_local(inputs)
					outputs = torch.log_softmax(outputs, dim=1)
					loss = criterion(outputs, labels)
				

				if torch.cuda.is_available():					
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
			metrics = evaluate(idx_to_supergenre, model_local, val_dataloader, criterion, device, shortest_graph, train_dataset)#, epoch=epoch)
		
			# Early stopping based on mean distance
			if metrics['mean_distance'] < best_mean_distance:
				best_mean_distance = metrics['mean_distance']
				best_model_state = model_local.state_dict()
				counter = 0
				print(f"[Epoch {epoch+1}] New best mean_distance: {best_mean_distance:.4f}")
			else:
				counter += 1
				print(f"No improvement in mean_distance for {counter} epoch(s).")
		
			# SWA Averaging
			if epoch >= swa_start_epoch:
				swa_model.update_parameters(model_local)
		
			# Append metrics to lists
			for lst, key in zip(
				[val_loss_list, val_accuracy_list, top3_acc_list, top5_acc_list, precision_list, recall_list, f1_list,
				mean_dist_list, std_dist_list, super_acc_list, super_top3_acc_list],
				["val_loss", "val_accuracy", "top3_accuracy", "top5_accuracy", "precision", "recall", "f1",
				"mean_distance", "std_distance", "supergenre_accuracy", "supergenre_top3_accuracy"]
			):
				lst.append(metrics[key])
		
			# Print metrics table
			headers = ["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc",
					"Top-3 Acc", "Top-5 Acc", "Precision", "Recall", "F1",
					"Supergenre Acc", "Top-3 Super", "Mean Dist", "Std Dist"]
		
			row = [epoch + 1,
				f"{avg_train_loss:.4f}", f"{train_accuracy:.2%}",
				f"{metrics['val_loss']:.4f}", f"{metrics['val_accuracy']:.2%}",
				f"{metrics['top3_accuracy']:.2%}", f"{metrics['top5_accuracy']:.2%}",
				f"{metrics['precision']:.2%}", f"{metrics['recall']:.2%}", f"{metrics['f1']:.2%}",
				f"{metrics['supergenre_accuracy']:.2%}", f"{metrics['supergenre_top3_accuracy']:.2%}",
				f"{metrics['mean_distance']:.4f}", f"{metrics['std_distance']:.4f}"]
		
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
		
		# Save the best model
		if best_model_state is not None:
			torch.save(best_model_state, output_dir/'best_model.pth')
			print(f"Best model saved with mean_distance: {best_mean_distance:.4f}")
		
		# Save the SWA model
		torch.save(swa_model.module.state_dict(), output_dir/'swa_model.pth')
		print("SWA model saved from averaged checkpoints.")

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
			"mean_distance": mean_dist_list,
			"std_distance": std_dist_list
		})
		metrics_df.to_csv(output_dir / "training_metrics.csv", index=False)

		# Plot the metrics
		plot_metrics(val_accuracy_list, top3_acc_list, top5_acc_list, precision_list, recall_list, f1_list, mean_dist_list, super_acc_list, super_top3_acc_list, output_dir/"metrics.png" )

		# Save the final confusion matrices to the output directory
		cm_genre = metrics['genre_confusion_matrix']
		cm_genre.to_csv(output_dir / "confusion_matrix_genre.csv", index=True)

		cm_supergenre = metrics['supergenre_confusion_matrix']
		cm_supergenre.to_csv(output_dir / "confusion_matrix_supergenre.csv", index=True)

		# Also save these using percentages
		cm_genre_pct = cm_genre.div(cm_genre.sum(axis=1), axis=0) * 100
		cm_genre_pct.to_csv(output_dir / "confusion_matrix_genre_percent.csv", index=True)

		cm_supergenre_pct = cm_supergenre.div(cm_supergenre.sum(axis=1), axis=0) * 100
		cm_supergenre_pct.to_csv(output_dir / "confusion_matrix_supergenre_percent.csv", index=True)

		# Save a heatmap of the confusion matrices
		plot_cm_super(cm_supergenre, output_dir/"supergenre_confusion_matrix.png")
		plot_cm_super(cm_supergenre_pct, output_dir / "supergenre_confusion_matrix_percent.png", percent=True)
