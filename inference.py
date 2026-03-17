from data.dataset import SingleAudioSpectrogramDataset
from config import (
    EXPORT_DATA_PATH,
    SAMPLING_RATE
)
from torch.utils.data import DataLoader
import torch
import torch
from pathlib import Path
import numpy as np
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, FinerCAM
from models.cnn import CustomCNNModel
from models.vit import ViTModel
from config import SAMPLING_RATE, FIG_PATH
import librosa
import os
import matplotlib.pyplot as plt
import glob 
from pathlib import Path

# file_path = 'samples/loa_LmS_08;00_CHI_ph,dr_rt_tv.wav'

dir_name = "/net/projects/scratch/winter/valid_until_31_July_2026/ybrima/Datasets/new_ph_clips"
file_list = glob.glob(os.path.join(dir_name, "**/*.wav"), recursive=True)

idx = np.random.randint(0, len(file_list))
file_path = file_list[idx]
class_name = Path(file_path).parent.name

dataset = SingleAudioSpectrogramDataset(file_path, label=2, target_sample_rate=SAMPLING_RATE, root_dir=f"{EXPORT_DATA_PATH}/train")
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

X = next(iter(dataloader))

true_label = dataset.class_to_idx[class_name]


print(true_label, class_name, file_path)

print(f"Selected file: {file_path} from class: {class_name}")

# -----------------------------
# DEVICE
# -----------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# -----------------------------
# RESHAPE TRANSFORM FOR ViT
# -----------------------------
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# -----------------------------
# LOAD MODEL
# -----------------------------
def load_trained_model(model_name: str, model_path: str, num_classes: int, modelstr='resnet18'):
    if model_name == "CustomCNNModel":
        model = CustomCNNModel(num_classes=num_classes, modelstr=modelstr)
    elif model_name == "ViTModel":
        model = ViTModel(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# -----------------------------
# Single audio CAM visualization
# -----------------------------
def single_audio_cam_plot(
    model,
    audio_path: str,
    label: int,
    target_layers,
    model_name: str,
    n_fft=512,
    hop_length=256,
    power=2.0,
    normalize=True,
    transform=None,
    output_dir=FIG_PATH,
    save_fig=True
):
    dataset = SingleAudioSpectrogramDataset(
        file_path=audio_path,
        label=label,
        target_sample_rate=SAMPLING_RATE,
        n_fft=n_fft,
        hop_length=hop_length,
        power=power,
        root_dir=f"{EXPORT_DATA_PATH}/train",
        normalize=normalize,
        transform=transform
    )

    sample, true_label = dataset[0]
    waveform = sample['waveform']
    spectrogram = sample['data']
    sr = sample['sample_rate']

    input_tensor = spectrogram.unsqueeze(0).float().to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    print(f"Predicted class: {predicted_class} (confidence: {confidence:.2f}), True label: {true_label}")

    # CAM methods
    cam_methods = {
        "GradCAM": GradCAM,
        "GradCAM++": GradCAMPlusPlus,
        "ScoreCAM": ScoreCAM,
        "FinerCAM": FinerCAM
    }

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_cols = 2 + len(cam_methods)  # waveform, spectrogram, CAMs
    fig, axs = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))

    # Ensure axs is always iterable
    if n_cols == 1:
        axs = [axs]

    # -----------------------
    # Column 0: waveform
    # -----------------------
    waveform_clean = waveform.squeeze().cpu().numpy()
    axs[0].set_title("Waveform", fontsize=14)
    librosa.display.waveshow(waveform_clean, sr=sr, ax=axs[0])
    axs[0].set_xlabel("Time (s)", fontsize=12)
    axs[0].set_ylabel("Amplitude", fontsize=12)

    # -----------------------
    # Column 1: spectrogram
    # -----------------------
    S = librosa.stft(waveform_clean, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    time_frames = np.arange(S_db.shape[1])
    time_seconds = librosa.frames_to_time(time_frames, sr=sr, hop_length=hop_length)
    extent = [time_seconds[0], time_seconds[-1], 0, S_db.shape[0]]

    axs[1].imshow(S_db, origin='lower', aspect='auto', cmap='viridis', extent=extent)
    axs[1].set_title(f"Original Spectrogram\nTrue: {true_label}", fontsize=14)
    axs[1].set_xlabel("Time (s)", fontsize=12)
    axs[1].set_ylabel("Frequency (bins)", fontsize=12)

    # -----------------------
    # Columns 2+: CAM overlays
    # -----------------------
    for col_idx, (method_name, CAMClass) in enumerate(cam_methods.items(), start=2):
        if model_name == "ViTModel":
            cam = CAMClass(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
        else:
            cam = CAMClass(model=model, target_layers=target_layers)

        targets = [ClassifierOutputTarget(predicted_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

        # Resize CAM to match spectrogram
        if grayscale_cam.shape != S_db.shape:
            from scipy.ndimage import zoom
            scale_factors = (S_db.shape[0] / grayscale_cam.shape[0],
                             S_db.shape[1] / grayscale_cam.shape[1])
            grayscale_cam_resized = zoom(grayscale_cam, scale_factors, order=1)
        else:
            grayscale_cam_resized = grayscale_cam

        axs[col_idx].imshow(S_db, origin='lower', aspect='auto', cmap='viridis', extent=extent)
        axs[col_idx].imshow(grayscale_cam_resized, cmap='inferno', alpha=0.5, origin='lower', aspect='auto', extent=extent)
        axs[col_idx].set_title(f"{method_name}\nPred: {predicted_class}", fontsize=14)
        axs[col_idx].set_xlabel("Time (s)", fontsize=12)
        axs[col_idx].set_ylabel("Frequency (bins)", fontsize=12)

    plt.tight_layout()
    if save_fig:
        save_path = Path(output_dir) / f"Single_Individual_CAM_{class_name}.png"
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Saved CAM figure → {save_path}")
    else:
        plt.show()

    return predicted_class, confidence, true_label

if __name__ == "__main__":
    # Example usage
    model_path = "/net/projects/scratch/winter/valid_until_31_July_2026/ybrima/Models/ChimpSSL/best_model_experiment_resnet18_chimp_resnet18_exp_200.pth" #best_model_experiment_200.pth" #best_model_experiment_resnet18_chimp_resnet18_exp_200.pth"
    model_name = "CustomCNNModel"  # or "ViTModel"
    num_classes = 11  # Adjust based on your dataset
    modelstr = 'resnet18'  # Specify the model architecture if using CustomCNNModel
    model = load_trained_model(model_name, model_path, num_classes, modelstr=modelstr)

    target_layers = [model.base_model.layer4[-1]]  # For CustomCNNModel with ResNet backbone
    audio_path = file_path
    # true_label =  2 # replace with actual label

    pred_class, conf, true_label = single_audio_cam_plot(
        model, audio_path, true_label, target_layers, model_name, save_fig=True
    )
