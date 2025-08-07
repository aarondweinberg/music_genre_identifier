# Tuning ResNet18 for graph weight hyperparameters

# Features: mel pcen spectrograms
#			from 15-second clips, each augmented with pitch shift and pink noise
# Architecture: ResNet18 (modified for our inputs)
# Beta: 2
# Graph weights: tuning using optuna
# Inputs: A variety of spectrograms of dimension 12x256, 32x256, and 128x256 (also graphs of dimension 300x300)
#		  We'll adjust the ResNet18 architecture use stride=1 in conv1 and skip maxpool when the height is 12 or 32
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
#from torch.amp import GradScaler
from torch.amp import autocast
from torch.cuda.amp import GradScaler	# Depreciated, but the previous line is throwing an error
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
import optuna
import json
import random
import csv



############# === IMPORT YOUR MODULES ===
############# from transforms import get_transform	# optional

# Global Parameters

# Base directories for train and val
train_data_directory = "./15_second_features_augmented/musicmap_processed_output_splits_train/mel_pcen_gray"
val_data_directory = "./15_second_features_augmented/musicmap_processed_output_splits_val/mel_pcen_gray"

feature_dir = Path("features_mel_pcen")
output_root = Path("optuna_weights")

# GradScaler might speed up cuda, but is not available on mps
scaler = GradScaler(enabled=torch.cuda.is_available())	# This version is deprecated, but the other is throwing errors...

#scaler = GradScaler(
#	enabled=torch.cuda.is_available(),
#	device_type='cuda' if torch.cuda.is_available() else 'cpu'
#)
beta = 2

#random.seed(42)
#np.random.seed(42)
#torch.manual_seed(42)



# === MAIN ENTRYPOINT ===
def main(args):
	# Set working directory and sys.path
	if args.dir:
		sys.path.append(args.dir)
		os.chdir(args.dir)

	# Local Imports
	from musicmap_graph_creator import create_musicmap_graph, compute_shortest_paths_between_classes_nx
	from model_resnet18flex import ResNet18Gray
	from musicmap_helpers import compute_image_dataset_stats, check_set_gpu, batch_soft_labeling_loss, top_k_accuracy, evaluate, plot_metrics, plot_cm_super, get_input_height

	# Set the device to the local GPU if available
	# Need to do this inside main because it's not inside the optuna function
	device = check_set_gpu()


	# === OBJECTIVE FUNCTION ===
	def objective(trial):
		# --- Suggest edge weights ---
		primary = trial.suggest_int("primary", 1, 2)
		secondary = trial.suggest_int("secondary", 1, 2)
		backlash = trial.suggest_int("backlash", 1, 3)
		supergenre = trial.suggest_int("supergenre", 1, 4)
		world_supergenre = trial.suggest_categorical("world_supergenre", [1])
		world_cluster = trial.suggest_categorical("world_cluster", [1])
		util_genre = trial.suggest_int("util_genre", 1, 2)
		cluster = trial.suggest_categorical("cluster", [1])
	
		combo = (primary, secondary, backlash, supergenre,
				 world_supergenre, world_cluster, util_genre, cluster)
		trial.set_user_attr("edge_combo", combo)

		# Print the trial params to the text buffer
		params = trial.params
		print(f"Trial {trial.number} with params: {params}")
	
		# Set the output directory
		output_dir = output_root / f"trial_{trial.number}"
		output_dir.mkdir(parents=True, exist_ok=True)
	
		# Set up a log file
		log_file_path = output_dir / "epoch_metrics_log.csv"
		log_file = open(log_file_path, mode="w", newline="")
		csv_writer = csv.writer(log_file)
		csv_writer.writerow([
			"epoch", "primary", "secondary", "backlash",
			"supergenre", "world_supergenre", "world_cluster", "util_genre", "cluster",
			"avg_train_loss", "train_accuracy",
			"val_loss", "val_accuracy", "top3_accuracy", "top5_accuracy",
			"precision", "recall", "f1",
			"supergenre_accuracy", "supergenre_top3_accuracy",
			"mean_distance"
		])
	
		# --- Build graph ---
		musicmap_graph = create_musicmap_graph(
			primary_edge_weight=primary,
			secondary_edge_weight=secondary,
			backlash_edge_weight=backlash,
			supergenre_edge_weight=supergenre,
			world_supergenre_weight=world_supergenre,
			world_cluster_weight=world_cluster,
			util_genre_weight=util_genre,
			cluster_weight=cluster
		)
		shortest_graph, class_names = compute_shortest_paths_between_classes_nx(
			class_dir=train_data_directory,	 # path to folder with class folders
			graph=musicmap_graph,
			return_tensor=True
		)
	
		###===Transform===
		# We'll use the means and standard deviations to normalize the images.
		
		image_dir = train_data_directory
		means, sds = compute_image_dataset_stats(image_dir)
		
		transform_greyscale = transforms.Compose([
			transforms.ToTensor(), # Should always be the last step before feeding into a model
			transforms.Grayscale(num_output_channels=1),  # force grayscale
			transforms.Normalize(mean=means, std=sds)	 # Normalize to imagenet mean and standard deviation
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
			batch_size=64,	# Adjust batch size as needed
			shuffle=True,
			num_workers=4,	# Adjust this to tweak multiprocessing
			pin_memory=True,
			prefetch_factor=2
		)
		
		val_dataloader = torch.utils.data.DataLoader(
			val_dataset,
			batch_size=10,	# Adjust batch size as needed
			shuffle=False, #Setting to false keeps evaluation stable across epochs
			num_workers=4,
			pin_memory=True,
			prefetch_factor=2 
		)
	
		# Create index-to-supergenre lookup once
		idx_to_supergenre = np.array([cls[:3] for cls in train_dataset.classes])  # shape (num_classes,)
	
		# --- Model, Optimizer, Loss ---
		num_classes = len(train_dataloader.dataset.classes) 
		
		input_height = get_input_height(train_data_directory)
		model = ResNet18Gray(num_classes=num_classes, input_height=input_height)
		model = model.to(device)
	
		optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
		criterion = lambda outputs, labels: batch_soft_labeling_loss(
			outputs, labels, shortest_graph, beta
		)
	
		# --- Train loop ---
	#	best_metric = float("inf") #use this if training on mean_distance
		best_metric = 0.0	# use this for maximizing (e.g.) accuracy
		best_state = None
		patience = 5
		counter = 0
		epochs = 25
	
		use_amp = torch.cuda.is_available()
	
		for epoch in range(epochs):
			model.train()
			running_loss = 0.0
			correct = 0
			total = 0
			for inputs, labels in train_dataloader:
				inputs, labels = inputs.to(device), labels.to(device)
				optimizer.zero_grad()
				
				
				if use_amp:
					with autocast(device_type='cuda'):
						outputs = model(inputs)
						outputs = torch.log_softmax(outputs, dim=1)
						loss = criterion(outputs, labels)
				else:
					outputs = model(inputs)
					outputs = torch.log_softmax(outputs, dim=1)
					loss = criterion(outputs, labels)
	
				if use_amp:
					scaler.scale(loss).backward()
					scaler.step(optimizer)
					scaler.update()
				else:
					loss.backward()
					optimizer.step()
	
	#			loss.backward()
	#			optimizer.step()
	
				running_loss += loss.item() * inputs.size(0)
				_, predicted = torch.max(outputs, 1)
				correct += (predicted == labels).sum().item()
				total += labels.size(0)
		
			avg_train_loss = running_loss / len(train_dataloader.dataset)
			train_accuracy = correct / total
	
			metrics = evaluate(idx_to_supergenre, model, val_dataloader, criterion, device, shortest_graph, train_dataset, use_amp)
	#		mean_distance = metrics["mean_distance"]
			val_accuracy = metrics["val_accuracy"]
	
			csv_writer.writerow([
				epoch,
				primary,
				secondary,
				backlash,
				supergenre,
				world_supergenre,
				world_cluster,
				util_genre,
				cluster,				
				round(avg_train_loss, 4),
				round(train_accuracy, 4),
				round(metrics["val_loss"], 4),
				round(metrics["val_accuracy"], 4),
				round(metrics["top3_accuracy"], 4),
				round(metrics["top5_accuracy"], 4),
				round(metrics["precision"], 4),
				round(metrics["recall"], 4),
				round(metrics["f1"], 4),
				round(metrics["supergenre_accuracy"], 4),
				round(metrics["supergenre_top3_accuracy"], 4),
				round(metrics["mean_distance"], 4)
			])
			log_file.flush()
			
			print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={metrics['val_loss']:.4f}, val_acc={metrics['val_accuracy']:.4f}, f1={metrics['f1']:.4f}")
	
			trial.report(val_accuracy, step=epoch)
	
			if val_accuracy > best_metric:
				best_metric = val_accuracy
				best_state = model.state_dict()
				counter = 0
			else:
				counter += 1
				if counter >= patience:
	#				trial.report(best_metric, epoch)
					raise optuna.TrialPruned()
	#				break
	
			scheduler.step(val_accuracy)
	
		# --- Save outputs ---
		torch.save(best_state, output_dir / "best_model.pt")
		with open(output_dir / "metrics.json", "w") as f:
			json.dump({k: v for k, v in metrics.items() if "confusion_matrix" not in k}, f, indent=2)
	
		log_file.close()
	
		return best_metric

	study = optuna.create_study(
		direction="maximize",	# use minimize if training on mean_distance
		study_name="edge_weight_tuning",
		storage=f"sqlite:///{args.study_file}",
		load_if_exists=True
		)
	study.optimize(objective, n_trials=args.n_trials)

	print("Best trial:")
	print("	 Value:", study.best_value)
	print("	 Params:", study.best_params)
	print("	 Edge combo:", study.best_trial.user_attrs["edge_combo"])
	
	with open("best_trial.json", "w") as f:
		json.dump({
			"value": study.best_value,
			"params": study.best_params,
			"edge_combo": study.best_trial.user_attrs["edge_combo"]
		}, f, indent=2)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--n_trials", type=int, default=50)
	parser.add_argument("--study_file", type=str, default="optuna_study.db")
	parser.add_argument("--dir", type=str, default=".")
	args = parser.parse_args()
	main(args)
