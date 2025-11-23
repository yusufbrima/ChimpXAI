import torch
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from data.dataset import AugSpectrogramDataset
from models.cnn import CustomCNNModel, ContrastiveCNN, FinetuningClassifier
from models.vit import ViTModel, ContrastiveViT
from config import EXPORT_DATA_PATH, RESULTS_PATH, MODELS_PATH, SAMPLING_RATE, FIG_PATH, DATA_SENTINEL
from scipy.ndimage import zoom
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, FinerCAM
from saliency import extract_one_sample_per_class,load_model

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# RESHAPE TRANSFORM FOR VIT
# -----------------------------
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result



# -----------------------------
# RUN SAMPLEWISE CAM + WAVEFORM
# -----------------------------
def run_cam_methods_samplewise_with_waveform(model, data_npz, output_dir, target_layers, model_name, modelstr_save_name, target_class,ft,contrastive_method, classes,target_layers_string,experiment=None,cfg=None):
    output_dir.mkdir(parents=True, exist_ok=True)

    cam_methods = {
        "GradCAM": GradCAM,
        "GradCAM++": GradCAMPlusPlus,
        "ScoreCAM": ScoreCAM,
        "FinerCAM": FinerCAM
    }

    data_array = data_npz['data']
    waveform_array = data_npz['waveform']
    sample_rate_array = data_npz['sample_rate']
    labels_array = data_npz['labels']

    n_samples = len(labels_array)
    n_methods = len(cam_methods)
    
    # Columns: waveform, original, CAMs
    n_cols = 2 + n_methods
    n_rows = n_samples

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))

    if n_rows == 1:
        axs = axs[None, :]
    if n_cols == 1:
        axs = axs[:, None]

    for row, (data, waveform, sr, label) in enumerate(zip(data_array, waveform_array, sample_rate_array, labels_array)):
        input_tensor = torch.tensor(data).unsqueeze(0).float().to(device)

        # Model prediction
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()

        waveform_clean = np.squeeze(waveform)
        n_fft = cfg['n_fft']
        hop_length = cfg['hop_length']

        # Compute spectrogram
        S = librosa.stft(waveform_clean, n_fft=n_fft, hop_length=hop_length)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

        # -----------------------
        # Column 0: waveform
        # -----------------------
        axs[row, 0].set_title("Waveform", fontsize=14)
        librosa.display.waveshow(waveform_clean, sr=sr, ax=axs[row, 0])
        axs[row, 0].set_xlabel("Time (s)", fontsize=14)
        axs[row, 0].set_ylabel("Amplitude", fontsize=14)
        axs[row, 0].tick_params(axis='both', which='major', labelsize=14)


        # -----------------------
        # Column 1: original spectrogram
        # -----------------------
        axs[row, 1].imshow(S_db, origin='lower', aspect='auto', cmap='viridis')
        axs[row, 1].set_title(f"Original\n True: {classes[label]}", fontsize=14)
        axs[row, 1].set_xlabel("Time (s)", fontsize=14)
        axs[row, 1].set_ylabel("Frequency (bins)", fontsize=14)
        axs[row, 1].tick_params(axis='both', which='major', labelsize=14)

        # -----------------------
        # Columns 2+: CAM overlays
        # -----------------------
        for col_idx, (method_name, CAMClass) in enumerate(cam_methods.items(), start=2):
            if model_name == "ViTModel":
                cam = CAMClass(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
            else:
                cam = CAMClass(model=model, target_layers=target_layers)

            targets = [ClassifierOutputTarget(label)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

            # print(f"Generated grayscale_cam shape: {grayscale_cam.shape} for method: {method_name} ")

            if grayscale_cam.ndim == 3:
                grayscale_cam = grayscale_cam[:, :, 0]

            # Resize CAM to match spectrogram
            if grayscale_cam.shape != S_db.shape:
                scale_factors = (S_db.shape[0] / grayscale_cam.shape[0],
                                 S_db.shape[1] / grayscale_cam.shape[1])
                grayscale_cam_resized = zoom(grayscale_cam, scale_factors, order=1)
            else:
                print("No resizing needed for CAM.")
                grayscale_cam_resized = grayscale_cam

            # ax.imshow(image_list[sentinel].squeeze().cpu().numpy(), origin="lower", aspect="auto", interpolation="nearest", cmap="viridis")
            # ax.imshow(cam_output_list[sentinel].squeeze().cpu().numpy(), cmap='magma', alpha=0.5, aspect='auto', origin='lower')

            axs[row, col_idx].imshow(S_db, origin='lower', aspect='auto', interpolation='nearest', cmap='viridis')
            axs[row, col_idx].imshow(grayscale_cam_resized, cmap='inferno', alpha=0.5, aspect='auto', origin='lower')
            axs[row, col_idx].set_title(f"{method_name}\nPred: {classes[predicted_class]} ({confidence:.2f})", fontsize=14)
            axs[row, col_idx].set_xlabel("Time (s)", fontsize=14)
            axs[row, col_idx].set_ylabel("Frequency (bins)", fontsize=14)
            # set the ticks fontsize
            axs[row, col_idx].tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    # Re-apply tick sizes after tight_layout
    for ax in axs.flat:
        ax.tick_params(axis='both', which='major', labelsize=14)

    if target_layers_string != '':
        target_layers_string = f"_{target_layers_string}"
    else:
        target_layers_string = ''
    if experiment is None:
        experiment = ''
    else:
        experiment = f"_exp_{experiment}"
    if not ft:
        save_path = output_dir / f"{target_class}_{modelstr_save_name}_samplewise_CAMs_waveform{target_layers_string}{experiment}.png"
    else:
        save_path = output_dir / f"{target_class}_{modelstr_save_name}_finetuned_{contrastive_method}_samplewise_CAMs_waveform{target_layers_string}{experiment}.png"
    # save_path = output_dir / f"{target_class}_{modelstr_save_name}_samplewise_CAMs_waveform.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved samplewise CAM+waveform figure → {save_path}")

# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=int, default=100)
    parser.add_argument('--model_name', type=str, default='CustomCNNModel')
    parser.add_argument('--modelstr', type=str, default='resnet18')
    parser.add_argument('--target_class', type=str, default=DATA_SENTINEL)
    parser.add_argument('--ft', type=str, default='false', help="finetuned flag: true/false")
    parser.add_argument('--contrastive_method', type=str, default='supcon')
    parser.add_argument('--target_layers_string', type=str, default='', help="Comma-separated layer names for CAM")
    parser.add_argument('--selector_idx', type=int, default=0, help="Index to select target layer from candidates")

    args = parser.parse_args()

    args.ft = args.ft.lower() == 'true'
    output_dir = Path(FIG_PATH)
    sample_file = Path(f"{RESULTS_PATH}/saliency_samples_experiment_{args.target_class}_{args.experiment}.npz")

    model, test_dataset, target_layers, cfg = load_model(args)
    classes = test_dataset.classes
    # Load or extract samples
    if not sample_file.exists():
        extract_one_sample_per_class(test_dataset, sample_file)

    data_npz = np.load(sample_file, allow_pickle=True)

    if args.model_name == 'CustomCNNModel':
        modelstr_save_name = args.modelstr
    elif args.model_name == 'ViTModel':
        modelstr_save_name = f'ViTModel_{args.modelstr}'
    else:
        raise ValueError(f"Unknown model_name: {args.model_name}")

    run_cam_methods_samplewise_with_waveform(
        model, data_npz, output_dir, target_layers, 
        args.model_name, modelstr_save_name, args.target_class,args.ft,args.contrastive_method,classes,args.target_layers_string,args.experiment, cfg)

if __name__ == "__main__":
    main()
