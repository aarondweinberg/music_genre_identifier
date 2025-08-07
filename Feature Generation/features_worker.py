# features_worker.py
import os
import numpy as np
import librosa
from librosa.feature.rhythm import tempo
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path
import subprocess
import io
from PIL import Image
import matplotlib.cm as cm

created_dirs = set()

def ensure_dir(path):
    if path not in created_dirs:
        os.makedirs(path, exist_ok=True)
        created_dirs.add(path)

def apply_augmentations(y_seg, sr):
    """Return list of (suffix, augmented_signal) tuples."""
    pink_noise = librosa.util.normalize(librosa.effects.preemphasis(np.random.normal(0, 1, size=len(y_seg))))
    y_noise = librosa.util.normalize(y_seg + 0.005 * pink_noise)
    y_pitch = librosa.effects.pitch_shift(y_seg, sr=sr, n_steps=1)
    return [("", y_seg), ("_noise", y_noise), ("_pitch", y_pitch)]

def load_audio_ffmpeg(file_path, sr=22050):
    command = ["ffmpeg", "-i", file_path, "-f", "wav", "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1", "-"]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        audio_data, sample_rate = sf.read(io.BytesIO(result.stdout), dtype='float32')
        return audio_data, sample_rate
    except subprocess.CalledProcessError as e:
        print("FFmpeg Error:", e.stderr.decode())
        raise RuntimeError(f"ffmpeg failed on file: {file_path}")

def save_image(data, output_path, target_shape=(224, 224), cmap='viridis', grayscale=False):
    if grayscale:
        norm = 255 * (data - np.min(data)) / (np.max(data) - np.min(data))
        img = Image.fromarray(norm.astype(np.uint8))
    else:
        colormap = cm.get_cmap(cmap)
        norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        colored = (colormap(norm)[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(colored)
    img = img.resize(target_shape, Image.BICUBIC)
    img.save(output_path)

def process_file(args):
    file_path, feature_types, segment_length, output_dir = args

    y, sr = load_audio_ffmpeg(file_path, sr=22050)
    y, _ = librosa.effects.trim(y)
    genre_id = Path(file_path).parent.name
    filename = Path(file_path).stem
    duration = librosa.get_duration(y=y, sr=sr)
    duration_record = {'file': filename, 'genre_id': genre_id, 'duration': duration}
    segment_samples = segment_length * sr
    num_segments = len(y) // segment_samples
    features = []

    target_shapes = {
        'mel_db': (256, 128), 'mel_pcen': (256, 128),
        'mfcc': (256, 32), 'chroma': (256, 12),
        'chroma_cq': (256, 12), 'chroma_bs': (256, 12),
        'mfcc_plot': (256, 64),
        'hpss_median': (256, 128), 'hpss_mean': (256, 128)
    }

    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        y_seg_orig = y[start:end]
        if len(y_seg_orig) < segment_samples:
            continue

        for suffix, y_seg in apply_augmentations(y_seg_orig, sr):
            segment_id = f"{filename}_seg{i}{suffix}"
            base_name = f"{segment_id}.png"
            feature_row = {'file': filename, 'segment': segment_id, 'genre_id': genre_id}

            mel = librosa.feature.melspectrogram(y=y_seg, sr=sr)
            mel_db = librosa.power_to_db(mel)
            mel_pcen = librosa.pcen(mel * (2**31))

            # Save ResNet RGB mel spectrogram
            resnet_mel_dir = os.path.join(output_dir, 'resnet_mel_rgb', genre_id)
            ensure_dir(resnet_mel_dir)
            save_image(mel_pcen, os.path.join(resnet_mel_dir, base_name), grayscale=False)

            # Scalar features
            mfcc = librosa.feature.mfcc(S=mel_db, sr=sr, n_mfcc=13)
            chroma = librosa.feature.chroma_stft(y=y_seg, sr=sr)
            chroma_cq = librosa.feature.chroma_cqt(y=y_seg, sr=sr)
            chroma_bs = librosa.feature.chroma_cens(y=y_seg, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y_seg)[0].mean()
            rms = librosa.feature.rms(y=y_seg)[0].mean()
            tempo_val = tempo(y=y_seg, sr=sr)[0]
            tonnetz = librosa.feature.tonnetz(y=y_seg, sr=sr).mean(axis=1)
            spec_centroid = librosa.feature.spectral_centroid(y=y_seg, sr=sr)[0].mean()
            spec_bandwidth = librosa.feature.spectral_bandwidth(y=y_seg, sr=sr)[0].mean()
            spec_contrast = librosa.feature.spectral_contrast(y=y_seg, sr=sr).mean()
            spec_flatness = librosa.feature.spectral_flatness(y=y_seg)[0].mean()
            spec_rolloff = librosa.feature.spectral_rolloff(y=y_seg, sr=sr)[0].mean()

            feature_row.update({
                'zcr': zcr, 'rms': rms, 'tempo': tempo_val,
                'spec_centroid': spec_centroid,
                'spec_bandwidth': spec_bandwidth,
                'spec_contrast': spec_contrast,
                'spec_flatness': spec_flatness,
                'spec_rolloff': spec_rolloff
            })
            for j, t in enumerate(tonnetz):
                feature_row[f'tonnetz_{j}'] = t

            # Save all spectrogram-based features
            for kind, data, y_axis in [
                ('mel_db', mel_db, 'mel'), ('mel_pcen', mel_pcen, 'mel'),
                ('mfcc', mfcc, None), ('chroma', chroma, None),
                ('chroma_cq', chroma_cq, None), ('chroma_bs', chroma_bs, None)
            ]:
                for suffix, gray in [('rgb', False), ('gray', True)]:
                    key = f"{kind}_{suffix}"
                    if key not in feature_types:
                        continue  # skip this if it's not in the feature list
                    outdir = os.path.join(output_dir, feature_types[key], genre_id)
                    ensure_dir(outdir)
                    save_image(data, os.path.join(outdir, base_name),
                            target_shape=target_shapes[kind], grayscale=gray)


            # MFCC plot (line)
            outdir = os.path.join(output_dir, feature_types['mfcc_plot'], genre_id)
            ensure_dir(outdir)
            plt.figure(figsize=(3, 3), dpi=100)
            for j in range(mfcc.shape[0]):
                plt.plot(mfcc[j])
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(os.path.join(outdir, base_name), bbox_inches='tight', pad_inches=0)
            plt.close()

            # HPSS variants
            y_harm, y_perc = librosa.effects.hpss(y_seg)
            for method, func in [('median', np.median), ('mean', np.mean)]:
                hpss_spec = librosa.stft(func([y_harm, y_perc], axis=0))
                hpss_db = librosa.amplitude_to_db(np.abs(hpss_spec))
                outdir = os.path.join(output_dir, feature_types[f"hpss_{method}"], genre_id)
                ensure_dir(outdir)
                save_image(hpss_db, os.path.join(outdir, base_name),
                           target_shape=target_shapes[f"hpss_{method}"], grayscale=True)

            features.append(feature_row)

    return features, duration_record
