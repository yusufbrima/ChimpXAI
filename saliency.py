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
from models.cnn import CustomCNNModel
from models.vit import ViTModel
from config import EXPORT_DATA_PATH, RESULTS_PATH, MODELS_PATH, SAMPLING_RATE, FIG_PATH,DATA_SENTINEL
from scipy.ndimage import zoom
from collections import defaultdict
import random
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, FinerCAM

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# EXTRACT SAMPLES AND SAVE NPZ
# -----------------------------

def extract_one_sample_per_class(dataset, save_path, seed=None):
    """
    Randomly extracts one sample per class from a dataset and saves the result
    as a compressed NumPy archive (.npz).

    Args:
        dataset: A PyTorch-style dataset where dataset[i] returns (sample, label),
                 and each sample is a dict containing 'data', 'waveform', 'sample_rate'.
        save_path (str): Path to save the compressed .npz file.
        seed (int, optional): Random seed for reproducibility.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Group indices by class
    class_indices = defaultdict(list)
    classes = dataset.classes
    for i in range(len(dataset)):
        _, label = dataset[i]
        class_indices[label].append(i)

    # Pick one random index per class
    class_to_sample = {}
    for label, idxs in class_indices.items():
        random_idx = random.choice(idxs)
        sample, _ = dataset[random_idx]
        class_to_sample[label] = sample

    # Collect arrays
    data_list, waveform_list, sample_rate_list, labels_list = [], [], [], []

    for label, sample in sorted(class_to_sample.items()):
        data = sample['data'].numpy()
        waveform = sample['waveform'].numpy()
        sample_rate = sample['sample_rate']

        # Ensure channel dimension exists
        if data.ndim == 2:
            data = data[:, None, :]
        if waveform.ndim == 2:
            waveform = waveform[:, None, :]

        data_list.append(data)
        waveform_list.append(waveform)
        sample_rate_list.append(sample_rate)
        labels_list.append(label)

    # Convert to NumPy arrays
    data_array = np.stack(data_list)
    waveform_array = np.stack(waveform_list)
    sample_rate_array = np.array(sample_rate_list)
    labels_array = np.array(labels_list)

    # Save compressed array
    np.savez_compressed(
        save_path,
        data=data_array,
        waveform=waveform_array,
        sample_rate=sample_rate_array,
        labels=labels_array
    )

    print(f"Saved {len(labels_array)} random class samples → {save_path}")

# -----------------------------
# RESHAPE TRANSFORM FOR VIT
# -----------------------------
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# -----------------------------
# LOAD MODEL
# -----------------------------
def load_model(args):
    if args.model_name == 'CustomCNNModel':
        modelstr_save_name = args.modelstr
    elif args.model_name == 'ViTModel':
        modelstr_save_name = 'ViTModel'
    else:
        raise ValueError(f"Unknown model_name: {args.model_name}")
    
    if not args.ft:
        model_path = Path(f"{MODELS_PATH}/best_model_experiment_{args.modelstr}_{args.target_class}_{modelstr_save_name}_exp_{args.experiment}.pth")
    else:
        Path(f"{MODELS_PATH}/finetuned_model_{args.modelstr}_{args.target_class}_{args.contrastive_method}_{modelstr_save_name}_exp_{args.experiment}.pth")

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    cfg_path = Path(f'{RESULTS_PATH}/best_hyperparams_experiment_{modelstr_save_name}_{args.experiment}_{args.target_class}.json')

    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    test_dataset = AugSpectrogramDataset(
        f"{EXPORT_DATA_PATH}/test",
        duration=cfg["duration"],
        target_sample_rate=SAMPLING_RATE,
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"]
    )
    num_classes = len(test_dataset.classes)

    # Load model
    if args.model_name == "CustomCNNModel":
        if args.modelstr == "resnet18" or args.modelstr == "resnet34" or args.modelstr == "resnet50":
            model = CustomCNNModel(num_classes=num_classes, modelstr=args.modelstr).to(device)
        elif args.modelstr == "dense121":
            model = CustomCNNModel(num_classes=num_classes, modelstr="dense121").to(device)
        else:
            raise ValueError(f"Unknown modelstr {args.modelstr}")
    elif args.model_name == "ViTModel":
        model = ViTModel(model_name="vit_base_patch16_224", num_classes=num_classes, pretrained=False, in_chans=1).to(device)
    else:
        raise ValueError(f"Unknown model_name {args.model_name}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    selector_idx = args.selector_idx
    # Select target layer
    if args.model_name == "CustomCNNModel":
        if args.modelstr == "resnet18" or args.modelstr == "resnet34" or args.modelstr == "resnet50":
            # target_layers = [model.base_model.layer4[-1].conv2]
            candidate_layers = [
                    model.base_model.layer3[-1],          # Mid-level
                    model.base_model.layer4[-1],          # Deepest block (default)
                    model.base_model.layer4[-1].conv2     # Last conv in layer4
                ]
            target_layers = [candidate_layers[selector_idx]]  # Remove .conv2
        elif args.modelstr == "dense121":
            candidate_layers = [
                model.base_model.features.denseblock3,       # Mid-level
                model.base_model.features.denseblock4,       # Deepest block (default)
                model.base_model.features.denseblock4.denselayer16  # Last dense layer
            ]
            target_layers = [candidate_layers[selector_idx]]  
    elif args.model_name == "ViTModel":
        candidate_layers = [
                model.model.blocks[6].norm1,       # Middle attention block
                model.model.blocks[-1].norm1,      # Last attention block (default)
                model.model.norm                    # Final normalization before classifier
            ]
        target_layers = [candidate_layers[selector_idx]]
    print(f"Loaded model: {args.model_name} from {model_path}")
    return model, test_dataset, target_layers,cfg

# -----------------------------
# RUN CLASSWISE CAM PLOTS
# -----------------------------
def run_cam_methods_classwise(model, data_npz, output_dir, target_layers, model_name,modelstr_save_name,target_class,classes,target_layers_string,experiment=None,cfg=None):
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

    n_classes = len(labels_array)
    # Get parameters from config
    n_fft = cfg['n_fft']
    hop_length = cfg['hop_length']

    for method_name, CAMClass in cam_methods.items():
        fig, axs = plt.subplots(2, n_classes, figsize=(4*n_classes, 6))

        for col, (data, waveform, sr, label) in enumerate(zip(data_array, waveform_array, sample_rate_array, labels_array)):
            # Prepare input tensor
 
            input_tensor = torch.tensor(data).unsqueeze(0).float().to(device)
            
            # Get model prediction
            model.eval()
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()


            # CAM
            if model_name == "ViTModel":
                cam = CAMClass(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
            else:
                cam = CAMClass(model=model, target_layers=target_layers)

            targets = [ClassifierOutputTarget(label)]  # Use the actual label
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

            # Handle if CAM returns RGB
            if grayscale_cam.ndim == 3:
                grayscale_cam = grayscale_cam[:, :, 0]
            
            # =====================================================
            # CREATE VISUAL SPECTROGRAM FROM ORIGINAL WAVEFORM
            # =====================================================
            waveform_clean = np.squeeze(waveform)
            
            # Compute spectrogram with same parameters as dataset
            S = librosa.stft(waveform_clean, n_fft=n_fft, hop_length=hop_length)
            S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
            
            # Resize CAM to match spectrogram dimensions if needed
            if grayscale_cam.shape != S_db.shape:
                scale_factors = (S_db.shape[0] / grayscale_cam.shape[0], 
                               S_db.shape[1] / grayscale_cam.shape[1])
                grayscale_cam_resized = zoom(grayscale_cam, scale_factors, order=1)
            else:
                grayscale_cam_resized = grayscale_cam

            # =====================================================
            # DISPLAY SPECTROGRAM WITH CAM OVERLAY
            # =====================================================
            # Display spectrogram
            axs[0, col].imshow(S_db, origin="lower", aspect="auto", interpolation="nearest", cmap="viridis")
            axs[0, col].imshow(grayscale_cam_resized, cmap='inferno', alpha=0.5, aspect='auto', origin='lower')

            # Fix x-axis to show time in seconds
            n_frames = S_db.shape[1]
            times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)
            # Set tick positions and labels
            tick_frames = np.linspace(0, n_frames - 1, 5).astype(int)  # 5 ticks
            tick_times = librosa.frames_to_time(tick_frames, sr=sr, hop_length=hop_length)
            axs[0, col].set_xticks(tick_frames)
            axs[0, col].set_xticklabels([f'{t:.2f}' for t in tick_times])

            # Title with true label and prediction
            correct = "✓" if predicted_class == label else "✗"
            title = f"True: {classes[label]} | Pred: {classes[predicted_class]} {correct}\nConf: {confidence:.2f}"
            axs[0, col].set_title(title, fontsize=14)

            axs[0, col].set_xlabel("Time (s)", fontsize=14)
            axs[0, col].set_ylabel("Frequency (bins)", fontsize=14)
            # set the ticks fontsize
            axs[0, col].tick_params(axis='both', which='major', labelsize=14)


            # Waveform plot
            axs[1, col].set_title("Waveform", fontsize=14)
            librosa.display.waveshow(np.squeeze(waveform), sr=sr, ax=axs[1, col])
            axs[1, col].set_xlabel("Time (s)", fontsize=14)
            axs[1, col].set_ylabel("Amplitude", fontsize=14)
            # set the ticks fontsize
            axs[1, col].tick_params(axis='both', which='major', labelsize=14)

        # plt.suptitle(f"{method_name} - All Classes", fontsize=14)
        plt.tight_layout()
        if target_layers_string != '':
            target_layers_string = f"_{target_layers_string}"
        else:
            target_layers_string = ''
        if experiment is None:
            experiment = ''
        else:
            experiment = f"_exp_{experiment}"
        save_path = output_dir / f"{target_class}_{modelstr_save_name}_{method_name}_all_classes{target_layers_string}{experiment}.png"
        plt.tight_layout()
        for ax in axs.flat:
            ax.tick_params(axis='both', which='major', labelsize=14)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f"Saved all classwise CAM+waveform figures → {output_dir}")

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

    model, test_dataset, target_layers,cfg = load_model(args)

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

    # {modelstr}_{target_class}_{modelstr_save_name}_{method_name}
    run_cam_methods_classwise(model, data_npz, output_dir, target_layers, args.model_name,modelstr_save_name,args.target_class,classes,args.target_layers_string,args.experiment,cfg)

if __name__ == "__main__":
    main()