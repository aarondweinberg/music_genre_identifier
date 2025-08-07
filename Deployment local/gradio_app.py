import os
import torch
import librosa
import gradio as gr
import numpy as np
from torchvision import transforms
from pathlib import Path
from PIL import Image
from model_resnet18flex_dual import DualBranchResNet18Gray
from musicmap_helpers import check_set_gpu, get_input_height

# === Load model ===
device = check_set_gpu()
model = DualBranchResNet18Gray(input_height1=128, input_height2=32, num_classes=250)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval().to(device)

# === Load shortest graph + class names ===
shortest_graph = torch.load("shortest_graph.pt")
with open("class_names.txt") as f:
    idx_to_class = [line.strip() for line in f]

beta = 2.5

# === Transforms ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def extract_segment_features(y, sr, start, end):
    y_seg = y[int(start * sr):int(end * sr)]
    if len(y_seg) < sr * 15:
        return None, None
    mel = librosa.feature.melspectrogram(y=y_seg, sr=sr)
    mel_pcen = librosa.pcen(mel * (2**31))
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), sr=sr, n_mfcc=13)

    mel_img = Image.fromarray(((mel_pcen - mel_pcen.min()) / mel_pcen.ptp() * 255).astype(np.uint8))
    mel_img = mel_img.resize((256, 128))
    mfcc_img = Image.fromarray(((mfcc - mfcc.min()) / mfcc.ptp() * 255).astype(np.uint8))
    mfcc_img = mfcc_img.resize((256, 32))

    return transform(mel_img), transform(mfcc_img)

def predict_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    y, _ = librosa.effects.trim(y)
    segment_len = 15
    total_secs = int(librosa.get_duration(y=y, sr=sr))
    logits_all = []

    for start in range(0, total_secs, segment_len):
        end = start + segment_len
        mel_tensor, mfcc_tensor = extract_segment_features(y, sr, start, end)
        if mel_tensor is None:
            continue
        input1 = mel_tensor.unsqueeze(0).to(device)
        input2 = mfcc_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input1, input2)
            logits_all.append(logits.cpu().numpy())

    if not logits_all:
        return "No valid segments found"

    avg_logits = np.mean(logits_all, axis=0)  # shape: (1, num_classes)
    scores = avg_logits[0]
    top5_idx = np.argsort(scores)[::-1][:5]
    genre_preds = [(idx_to_class[i], float(scores[i])) for i in top5_idx]

    # === SUPERGENRE ===
    supergenre_scores = {}
    for i, class_name in enumerate(idx_to_class):
        supergenre = class_name[:3]
        supergenre_scores.setdefault(supergenre, 0.0)
        supergenre_scores[supergenre] += scores[i]
    
    sorted_super = sorted(supergenre_scores.items(), key=lambda x: x[1], reverse=True)
    supergenre_preds = [(s, float(p)) for s, p in sorted_super[:5]]

    return {
        "Genre Prediction (Top 5)": {g: f"{s:.2%}" for g, s in genre_preds},
        "Supergenre Prediction": {s: f"{p:.2%}" for s, p in supergenre_preds}
    }

# === Gradio Interface ===
demo = gr.Interface(
    fn=predict_audio,
    inputs=gr.Audio(type="filepath", label="Upload audio file (15s+)"),
    outputs=[
        gr.Label(label="Genre"),
        gr.Label(label="Supergenre"),
    ],
    title="Music Genre and Supergenre Classifier",
    description="Upload a 15+ second audio file to get top-5 genre predictions and supergenre predictions using an ensemble of 15s segments."
)

if __name__ == "__main__":
    demo.launch()
