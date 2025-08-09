import os
import torch
import librosa
import gradio as gr
import numpy as np
from pathlib import Path
import pandas as pd
import networkx as nx
from model_resnet18flex_dual import DualBranchResNet18Gray
from musicmap_helpers import check_set_gpu, get_input_height
from torchvision import transforms
import torch.nn.functional as F
from collections import Counter
import soundfile as sf
from torchvision import transforms

# Load shortest graph (as .pt)
shortest_graph = torch.load("shortest_graph.pt", map_location="cpu")
if isinstance(shortest_graph, dict):
	G = nx.Graph()
	for node, neighbors in shortest_graph.items():
		for neighbor in neighbors:
			G.add_edge(node, neighbor)
	shortest_graph = G

# Load class names
with open("class_names.txt") as f:
	class_names = [line.strip() for line in f]
supergenres = sorted(set(name[:3] for name in class_names))

# Load code-to-label map
genre_labels = pd.read_csv("musicmap_genres.csv")
code_to_label = dict(zip(genre_labels["Code"], genre_labels["Label"]))

# Need to compute means and sds separately from the training data
mel_mean = torch.tensor([0.09798076748847961])
mel_std = torch.tensor([0.10669851303100586])
mfcc_mean = torch.tensor([0.6104416847229004])
mfcc_std = torch.tensor([0.13878336548805237])

normalize_mel = transforms.Normalize(mean=mel_mean, std=mel_std)
normalize_mfcc = transforms.Normalize(mean=mfcc_mean, std=mfcc_std)

# A function to norm the tensor
def minmax_scale(t):
    return (t - t.min()) / (t.max() - t.min() + 1e-8)

def preprocess_segment(seg, sr):
	# MEL PCEN (256x128)
	mel = librosa.feature.melspectrogram(y=seg, sr=sr, n_mels=128)
	mel_pcen = librosa.pcen(mel * (2 ** 31))
	print(f"Segment stats - mel min: {mel.min():.2f}, max: {mel.max():.2f}, PCEN min: {mel_pcen.min():.2f}, max: {mel_pcen.max():.2f}")
	mel_img = torch.tensor(mel_pcen).float()
	mel_img = minmax_scale(mel_img)
	mel_img = F.interpolate(mel_img.unsqueeze(0).unsqueeze(0), size=(128, 256), mode='bilinear', align_corners=False).squeeze()
	mel_img = mel_img.unsqueeze(0)  # shape: (1, 128, 256)
	mel_img = normalize_mel(mel_img)

	# MFCC (256x32)
	mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13)
	mfcc_img = torch.tensor(mfcc).float()
	mfcc_img = minmax_scale(mfcc_img)
	mfcc_img = F.interpolate(mfcc_img.unsqueeze(0).unsqueeze(0), size=(32, 256), mode='bilinear', align_corners=False).squeeze()
	mfcc_img = mfcc_img.unsqueeze(0)	# shape: (1, 32, 256)
	mfcc_img = normalize_mfcc(mfcc_img)

	return mel_img.unsqueeze(0), mfcc_img.unsqueeze(0)	# add batch dimension



def predict_genre(audio_file):
	y, sr = librosa.load(audio_file, sr=22050)
	print(f"Processing file: {audio_file}")
	print(f"Loaded audio: {audio_file}, duration: {len(y)/sr:.2f} sec, sr: {sr}")
	print(f"Waveform mean: {np.mean(y):.4f}, std: {np.std(y):.4f}, max: {np.max(np.abs(y)):.4f}")
	y, _ = librosa.effects.trim(y)	# remove silence
	segment_length = 15 * sr
	if len(y) < segment_length:
		raise ValueError("Audio too short after trimming silence.")
	segments = [y[i:i + segment_length] for i in range(0, len(y) - segment_length + 1, segment_length)]
	
	# Load the model (inside the function helps prevent preservation of internal state)
	model = DualBranchResNet18Gray(input_height1 = 128, input_height2 = 32, num_classes=len(class_names))
	state_dict = torch.load("best_model.pth", map_location="cpu")
	missing, unexpected = model.load_state_dict(state_dict, strict=False)
	print("Missing keys:", missing)
	print("Unexpected keys:", unexpected)
	model.load_state_dict(state_dict)
	print("Max absolute weight in final layer:", model.fc.weight.abs().max().item())
	print("Mean weight in final layer:", model.fc.weight.mean().item())

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()
	print("Model in eval mode:", not model.training)

	genre_votes = []
	supergenre_votes = []

	with torch.no_grad():
		for i, seg in enumerate(segments):
			mel_tensor, mfcc_tensor = preprocess_segment(seg, sr)
			mel_tensor = mel_tensor.to(device)
			mfcc_tensor = mfcc_tensor.to(device)

			if mel_tensor.shape[-2:] != (128, 256) or mfcc_tensor.shape[-2:] != (32, 256):
				print(f"Skipping invalid segment shape: mel {mel_tensor.shape}, mfcc {mfcc_tensor.shape}")
				continue

			logits = model([mel_tensor, mfcc_tensor])[0].cpu()
			#probs_seg = torch.nn.functional.softmax(logits, dim=0)
			log_probs_seg = torch.nn.functional.log_softmax(logits, dim=0)
			probs_seg = log_probs_seg.exp()
			print("Logits (first 10):", logits[:10].detach().numpy().round(3))
			print("Predicted index:", logits.argmax().item())

			top3_seg = torch.topk(probs_seg, k=3)
			seg_top3_indices = top3_seg.indices.tolist()
			seg_top3_codes = [class_names[idx] for idx in seg_top3_indices]
			seg_top3_labels = [code_to_label.get(code, code) for code in seg_top3_codes]
			seg_top3_scores = top3_seg.values.tolist()

			# Log top-3 per segment
			print(f"[Segment {i+1}] Top 3 predictions:")
			for label, code, score in zip(seg_top3_labels, seg_top3_codes, seg_top3_scores):
				print(f"  - {label} ({code}): {round(score, 4)}")

			# Add top-1 genre and supergenre to vote lists
			top1_code = seg_top3_codes[0]
			genre_votes.append(top1_code)
			supergenre_votes.append(top1_code[:3])	# assumes first 3 letters are supergenre code

	# Majority vote
	genre_counts = Counter(genre_votes)
	top3_genres = genre_counts.most_common(3)
	top3_output = {
		f"{code_to_label.get(code, code)} ({code})": round(count / len(genre_votes), 4)
		for code, count in top3_genres
	}

	supergenre_counts = Counter(supergenre_votes)
	top3_supergenres = supergenre_counts.most_common(3)
	top3_supergenres_output = {
		f"{code_to_label.get(sg, sg)} ({sg})": round(count / len(supergenre_votes), 4)
		for sg, count in top3_supergenres
	}

	# Neighbors of top-1 genre
	top1 = top3_genres[0][0]
	print(f"\nTop-1 genre code: {top1}")
	if top1 in shortest_graph:
		neighbors = list(shortest_graph.neighbors(top1))
		print(f"Neighbors of {top1}: {neighbors}")
	else:
		print(f"{top1} not in shortest_graph")
		neighbors = []
	neighbor_output = {
		f"{code_to_label.get(code, code)} ({code})": round(genre_counts.get(code, 0) / len(genre_votes), 4)
		for code in sorted(neighbors)[:5]
	}
	print(f"Neighbor votes: {neighbor_output}")
	
	model.cpu()
	torch.cuda.empty_cache()

	return top3_output, top3_supergenres_output, neighbor_output


# Gradio interface
demo = gr.Interface(
	fn=predict_genre,
	inputs=gr.Audio(type="filepath", label="Upload Audio (MP3/M4A)"),
	outputs=[
		gr.Label(label="Top-3 Genre Predictions"),
		gr.Label(label="Top-3 Supergenre Predictions"),
		gr.Label(label="Distance-1 Neighbors of Top Genre")
	],
	title="Music Genre Classifier",
	description="Upload an audio clip (e.g., MP3 or M4A) of 'Western' music. The classifier will try to identify the genre, supergenre, and 'nearby' genres",
	cache_examples=False
)

if __name__ == "__main__":
	demo.launch()