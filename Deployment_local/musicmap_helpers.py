import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import pandas as pd
from PIL import Image
import os
from torch.cuda.amp import autocast

# Compute means and SDs from images to normalize inputs
def compute_image_dataset_stats(image_dir, batch_size=128, num_workers=4):
	transform = transforms.Compose([
		transforms.Grayscale(num_output_channels=1),  # Ensure 1 channel
		transforms.ToTensor(),
#		 transforms.Lambda(lambda x: x[:3, :, :] if x.shape[0] == 4 else x)	 # remove alpha if needed
	])

	dataset = datasets.ImageFolder(image_dir, transform=transform)
	loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

	n_images = 0
	mean = 0.0
	std = 0.0

	for batch, _ in loader:
		n = batch.size(0)
		mean += batch.mean(dim=[0, 2, 3]) * n
		std += batch.std(dim=[0, 2, 3]) * n
		n_images += n

	mean /= n_images
	std /= n_images

	return tuple(mean.tolist()), tuple(std.tolist())

# Check and set GPU availability
def check_set_gpu(override=None):
	if override is None:
		if torch.cuda.is_available():
			device = torch.device('cuda')
			print(f"Using GPU: {torch.cuda.get_device_name(0)}")
		elif torch.backends.mps.is_available():
			device = torch.device('mps')
			print(f"Using MPS: {torch.backends.mps.is_available()}")
		else:
			device = torch.device('cpu')
			print(f"Using CPU: {torch.device('cpu')}")
	else:
		device = torch.device(override)
	return device

#Here I'll try to define the soft labeling for a given shortest_paths tensor, following the Bertinetto paper. This is like cross-entropy,
#but rather than just including a term for the actual label for a given input, we include weighted terms for all labels,
#where the weights are determined by proximity to the true weight in the shortest_paths tensor.

def batch_soft_labeling_loss(log_probs, targets, shortest_paths, beta):
	"""
	Args:
		probs (Tensor): [batch_size, num_classes] predicted probabilities
		targets (Tensor): [batch_size] true class indices
		shortest_paths (Tensor): [num_classes, num_classes] class distance matrix
		beta (float): Softness parameter

	Returns:
		Tensor: scalar loss (mean over batch)
	"""
	batch_size, num_classes = log_probs.shape

	# Create weight matrix: [batch_size, num_classes]
	#distances = shortest_paths[targets]							 # [batch_size, num_classes]
	distances = shortest_paths.to(targets.device)[targets]			# Need to move these distances to the GPU
	weights = torch.exp(-beta * distances)							# [batch_size, num_classes]
	weights = weights / weights.sum(dim=1, keepdim=True)		 # Normalize rows

	# Cross-entropy with soft labels
	loss = -torch.sum(weights * log_probs, dim=1)				   # [batch_size]
	return loss.mean()											   # scalar


# A helper function for computing top_k_accuracy
def top_k_accuracy(output, target, k=3):
	with torch.no_grad():
		_, pred = output.topk(k, dim=1)
		return (pred == target.unsqueeze(1)).any(dim=1).float().mean().item()
	

# Evaluation function
# It takes the model, dataloader, criterion, device, shortest_graph, and epoch as inputs
# and returns a dictionary with evaluation metrics
# Note that this requires idx_to_supergenre array to have been defined in the training code
def evaluate(idx_to_supergenre, model, dataloader, criterion, device, shortest_graph, train_dataset, cm_csv_path=None, use_amp=False):
	model.eval()
	val_loss, correct, total = 0.0, 0, 0
	all_preds, all_labels = [], []

	with torch.no_grad():
		for inputs, labels in dataloader:
			inputs, labels = inputs.to(device), labels.to(device)
			
			if use_amp:
				with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
					outputs = model(inputs)
					outputs = torch.log_softmax(outputs, dim=1)
					loss = criterion(outputs, labels)
			else:
				outputs = model(inputs)
				outputs = torch.log_softmax(outputs, dim=1)
				loss = criterion(outputs, labels)

			val_loss += loss.item()
			_, predicted = outputs.max(1)
			total += labels.size(0)
			correct += predicted.eq(labels).sum().item()

			all_preds.append(outputs)
			all_labels.append(labels)


	# Combine all predictions and labels
	all_preds = torch.cat(all_preds)
	all_labels = torch.cat(all_labels)

	val_accuracy = correct / total
	avg_val_loss = val_loss / len(dataloader)

	# Move to CPU
	pred_classes = all_preds.argmax(dim=1).cpu()
	true_classes = all_labels.cpu()

	pred_classes_np = pred_classes.numpy()
	true_classes_np = true_classes.numpy()
	
	# Extract distances and compute mean and SD (may include inf values)
	#distances = shortest_graph[pred_classes, true_classes]
	distances = shortest_graph[pred_classes.cpu().numpy(), true_classes.cpu().numpy()]
	distances = torch.tensor(distances)  # convert back to Tensor for .mean()/.std()
	mean_shortest_distance = distances.float().mean().item()
	std_shortest_distance = distances.float().std().item()

	# Supergenre metrics
	pred_super = idx_to_supergenre[pred_classes_np]
	true_super = idx_to_supergenre[true_classes_np]
	supergenre_accuracy = np.mean(pred_super == true_super)

	# Top-3 supergenre accuracy
	top3_indices = torch.topk(all_preds, k=3, dim=1).indices.cpu().numpy()
	top3_supergenre_hits = sum(true_super[i] in idx_to_supergenre[top3] for i, top3 in enumerate(top3_indices))
	supergenre_top3_accuracy = top3_supergenre_hits / len(true_super)

	# Standard classification metrics
	precision = precision_score(true_classes_np, pred_classes_np, average='macro', zero_division=0)
	recall = recall_score(true_classes_np, pred_classes_np, average='macro', zero_division=0)
	f1 = f1_score(true_classes_np, pred_classes_np, average='macro', zero_division=0)

	# Top-k accuracies
	top3_acc = top_k_accuracy(all_preds, all_labels, k=3)
	top5_acc = top_k_accuracy(all_preds, all_labels, k=5)

	# Save confusion matrix
	cm = confusion_matrix(true_classes_np, pred_classes_np)
	class_names = train_dataset.classes
	cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

	# Compute supergenre confusion matrix
	unique_supergenres = np.unique(np.concatenate((true_super, pred_super)))
	cm_super = confusion_matrix(true_super, pred_super, labels=unique_supergenres)
	cm_super_df = pd.DataFrame(cm_super, index=unique_supergenres, columns=unique_supergenres)

	# Return all metrics as a dictionary
	return {
		"val_loss": avg_val_loss,
		"val_accuracy": val_accuracy,
		"top3_accuracy": top3_acc,
		"top5_accuracy": top5_acc,
		"precision": precision,
		"recall": recall,
		"f1": f1,
		"mean_distance": mean_shortest_distance,
		"std_distance": std_shortest_distance,
		"supergenre_accuracy": supergenre_accuracy,
		"supergenre_top3_accuracy": supergenre_top3_accuracy,
		"genre_confusion_matrix": cm_df,
		"supergenre_confusion_matrix": cm_super_df
	}

def evaluate_dual(idx_to_supergenre, model, dataloader, criterion, device, shortest_graph, train_dataset, cm_csv_path=None, use_amp=False):
	model.eval()
	val_loss, correct, total = 0.0, 0, 0
	all_preds, all_labels = [], []

	with torch.no_grad():
		for (input1, input2), labels in dataloader:
			input1, input2, labels = input1.to(device), input2.to(device), labels.to(device)
			
			if use_amp:
				with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
					outputs = model((input1, input2))
					outputs = torch.log_softmax(outputs, dim=1)
					loss = criterion(outputs, labels)
			else:
				outputs = model((input1, input2))
				outputs = torch.log_softmax(outputs, dim=1)
				loss = criterion(outputs, labels)

			val_loss += loss.item()
			_, predicted = outputs.max(1)
			total += labels.size(0)
			correct += predicted.eq(labels).sum().item()

			all_preds.append(outputs)
			all_labels.append(labels)

	# Combine predictions and labels
	all_preds = torch.cat(all_preds)
	all_labels = torch.cat(all_labels)

	val_accuracy = correct / total
	avg_val_loss = val_loss / len(dataloader)

	# Move to CPU
	pred_classes = all_preds.argmax(dim=1).cpu()
	true_classes = all_labels.cpu()

	pred_classes_np = pred_classes.numpy()
	true_classes_np = true_classes.numpy()
	
	# Compute shortest path distances
	distances = shortest_graph[true_classes_np, pred_classes_np]
	distances_tensor = torch.tensor(distances, dtype=torch.float32)
	mean_shortest_distance = distances_tensor.mean().item()
	std_shortest_distance = distances_tensor.std().item()

	# Supergenre metrics
	pred_super = idx_to_supergenre[pred_classes_np]
	true_super = idx_to_supergenre[true_classes_np]
	supergenre_accuracy = np.mean(pred_super == true_super)

	# Top-3 supergenre accuracy
	top3_indices = torch.topk(all_preds, k=3, dim=1).indices.cpu().numpy()
	top3_supergenre_hits = sum(true_super[i] in idx_to_supergenre[top3] for i, top3 in enumerate(top3_indices))
	supergenre_top3_accuracy = top3_supergenre_hits / len(true_super)

	# Standard metrics
	precision = precision_score(true_classes_np, pred_classes_np, average='macro', zero_division=0)
	recall = recall_score(true_classes_np, pred_classes_np, average='macro', zero_division=0)
	f1 = f1_score(true_classes_np, pred_classes_np, average='macro', zero_division=0)

	# Top-k accuracies
	top3_acc = top_k_accuracy(all_preds, all_labels, k=3)
	top5_acc = top_k_accuracy(all_preds, all_labels, k=5)

	# Confusion matrices
	cm = confusion_matrix(true_classes_np, pred_classes_np)
	class_names = train_dataset.classes
	cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

	unique_supergenres = np.unique(np.concatenate((true_super, pred_super)))
	cm_super = confusion_matrix(true_super, pred_super, labels=unique_supergenres)
	cm_super_df = pd.DataFrame(cm_super, index=unique_supergenres, columns=unique_supergenres)

	return {
		"val_loss": avg_val_loss,
		"val_accuracy": val_accuracy,
		"top3_accuracy": top3_acc,
		"top5_accuracy": top5_acc,
		"precision": precision,
		"recall": recall,
		"f1": f1,
		"mean_distance": mean_shortest_distance,
		"std_distance": std_shortest_distance,
		"supergenre_accuracy": supergenre_accuracy,
		"supergenre_top3_accuracy": supergenre_top3_accuracy,
		"genre_confusion_matrix": cm_df,
		"supergenre_confusion_matrix": cm_super_df
	}


def plot_metrics(val_accuracy_list, top3_acc_list, top5_acc_list, precision_list, recall_list, f1_list, mean_dist_list, super_acc_list, super_top3_acc_list, output_name="metrics.png" ):
	epochs = range(1, len(val_accuracy_list) + 1)

	plt.figure(figsize=(16, 12))

	# Validation Accuracy
	plt.subplot(3, 2, 1)
	plt.plot(epochs, val_accuracy_list, label='Top-1')
	plt.plot(epochs, top3_acc_list, label='Top-3')
	plt.plot(epochs, top5_acc_list, label='Top-5')
	plt.title('Validation Accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.legend()

	# Precision
	plt.subplot(3, 2, 2)
	plt.plot(epochs, precision_list, label='Precision', color='orange')
	plt.title('Validation Precision')
	plt.xlabel('Epoch')
	plt.ylabel('Score')

	# Recall
	plt.subplot(3, 2, 3)
	plt.plot(epochs, recall_list, label='Recall', color='green')
	plt.title('Validation Recall')
	plt.xlabel('Epoch')
	plt.ylabel('Score')

	# F1 Score
	plt.subplot(3, 2, 4)
	plt.plot(epochs, f1_list, label='F1 Score', color='red')
	plt.title('Validation F1 Score')
	plt.xlabel('Epoch')
	plt.ylabel('Score')

	# Mean Shortest Distance
	plt.subplot(3, 2, 5)
	plt.plot(epochs, mean_dist_list, label='Mean Shortest Distance', color='blue')
	plt.title('Mean Shortest Distance')
	plt.xlabel('Epoch')
	plt.ylabel('Distance')

	# Supergenre Accuracy
	plt.subplot(3, 2, 6)
	plt.plot(epochs, super_acc_list, label='Supergenre Accuracy', color='purple')
	plt.plot(epochs, super_top3_acc_list, label='Top-3 Supergenre Accuracy', color='brown')
	plt.title('Supergenre Accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.legend()

	plt.tight_layout()
	plt.savefig(output_name)
	plt.close()

def plot_cm_super(cm_super_df, output_name="supergenre_confusion_matrix.png", percent=False):
	plt.figure(figsize=(12, 10))
	if percent:
		sns.heatmap(cm_super_df, annot=True, fmt=".1f", cmap="Blues", vmin=0, vmax=100)
		plt.title("Supergenre Confusion Matrix (Percent)")
	else:
		sns.heatmap(cm_super_df, annot=True, fmt="d", cmap="Blues")
		plt.title("Supergenre Confusion Matrix")
	plt.ylabel("True Supergenre")
	plt.xlabel("Predicted Supergenre")
	plt.tight_layout()
	plt.savefig(output_name)
	plt.close()

def get_input_height(image_dir):
	"""
	Gets the height of the first image found in the directory tree.
	"""
	for root, _, files in os.walk(image_dir):
		for file in files:
			if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
				image_path = os.path.join(root, file)
				with Image.open(image_path) as img:
					return img.height
	raise RuntimeError(f"No image files found in {image_dir}")

class PairedFeatureDataset(torch.utils.data.Dataset):
	def __init__(self, root1, root2, transform1=None, transform2=None):
		self.dataset1 = datasets.ImageFolder(root=root1, transform=transform1)
		self.dataset2 = datasets.ImageFolder(root=root2, transform=transform2)

		assert self.dataset1.classes == self.dataset2.classes, "Mismatch in class folders"
		assert len(self.dataset1.samples) == len(self.dataset2.samples), "Mismatch in number of samples"

		self.classes = self.dataset1.classes
		self.transform1 = transform1
		self.transform2 = transform2

	def __getitem__(self, index):
		try:
			path1, label1 = self.dataset1.samples[index]
			path2, label2 = self.dataset2.samples[index]

			assert label1 == label2, f"Label mismatch at index {index}: {label1} != {label2}"

			img1 = Image.open(path1).convert("L")  # grayscale
			img2 = Image.open(path2).convert("L")

			if self.transform1:
				img1 = self.transform1(img1)
			if self.transform2:
				img2 = self.transform2(img2)

			return (img1, img2), label1
		except Exception as e:
			print(f"[Dataset Error] index={index}")
			print(f"  path1: {path1}, path2: {path2}")
			print(f"  label1: {label1}, label2: {label2}")
			raise e

	def __len__(self):
		return len(self.dataset1)