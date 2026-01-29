from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
# from dataset import AugSpectrogramDataset, Augmentation
from data.dataset import AugSpectrogramDataset,Augmentation
from models.cnn import SmallResCNNv5,CustomCNNModel
from models.vit import ViTModel
from config import (
    EXPORT_DATA_PATH,
    SAMPLING_RATE,
    RESULTS_PATH,
    MODELS_PATH,
    FT_EPOCHS,
    DATA_SENTINEL
)
from utils import train_model, test_model, EarlyStopping, compute_class_weights
import click
import json


# --- Device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@click.command()
@click.option('--experiment', default=200, type=int, help='Experiment number (matches the Optuna experiment)')
@click.option('--target_class', default=DATA_SENTINEL, help='Target class for classification')
@click.option('--model_name', default='CustomCNNModel', help='Model architecture name (e.g. SmallResCNNv5, CustomCNNModel, ViTModel)')
@click.option('--modelstr', default='resnet18', help='Model architecture to use if CustomCNNModel is selected')
def main(experiment, target_class, model_name, modelstr):

    # --- Load best hyperparameters from JSON ---
    # best_params_path = Path(f"{RESULTS_PATH}/best_hyperparams_experiment_{experiment}.json")
    if model_name == 'CustomCNNModel':
        modelstr_save_name = modelstr
    elif model_name == 'SmallResCNNv5':
        modelstr_save_name = 'SmallResCNNv5'
    elif model_name == 'ViTModel':
        modelstr_save_name = 'ViTModel'
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    best_params_path = Path(f'{RESULTS_PATH}/best_hyperparams_experiment_{modelstr_save_name}_{experiment}_{target_class}.json')
    if not best_params_path.exists():
        raise FileNotFoundError(f"No saved hyperparameters found at {best_params_path}")

    with open(best_params_path, 'r') as f:
        best_params = json.load(f)

    print("\nLoaded best hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # --- Apply loaded hyperparameters ---
    base_channels = best_params["base_channels"]
    dropout_p = best_params["dropout_p"]
    lr = best_params["lr"]
    batch_size = best_params["batch_size"]
    duration = best_params["duration"]
    n_fft = best_params["n_fft"]
    hop_length = best_params["hop_length"]

    augment = Augmentation(
        time_stretch=(best_params["time_stretch_rate"], best_params["time_stretch_rate"], 0.3),
        pitch_shift=(best_params["pitch_shift_semitones"], best_params["pitch_shift_semitones"], 0.3),
        shift_p=best_params["shift_prob"]
    )

    # --- Dataset setup ---
    train_ds = AugSpectrogramDataset(
        f"{EXPORT_DATA_PATH}/train",
        duration=duration,
        target_sample_rate=SAMPLING_RATE,
        transform=augment,
        n_fft=n_fft,
        hop_length=hop_length
    )

    val_dataset = AugSpectrogramDataset(
        f"{EXPORT_DATA_PATH}/val",
        duration=duration,
        target_sample_rate=SAMPLING_RATE,
        n_fft=n_fft,
        hop_length=hop_length
    )

    test_dataset = AugSpectrogramDataset(
        f"{EXPORT_DATA_PATH}/test",
        duration=duration,
        target_sample_rate=SAMPLING_RATE,
        n_fft=n_fft,
        hop_length=hop_length
    )

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # --- Class weights ---
    train_labels = [train_ds[i][1] for i in range(len(train_ds))]
    class_weights_eff = compute_class_weights(train_labels, method='effective', beta=0.99, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_eff)

    # --- Model setup ---
    num_classes = len(train_ds.classes)
    
    sample, label = next(iter(train_loader))

    input_shape = sample['data'].shape[1:]  

    # num_classes = len(train_dataset.classes)

    if model_name == "CustomCNNModel":
        model = CustomCNNModel(num_classes=num_classes, weights=None, modelstr=modelstr).to(device)
    elif model_name == "ViTModel":
        model =ViTModel(model_name='vit_base_patch16_224', num_classes=num_classes, pretrained=False, in_chans=1).to(device)
    else:
        model = SmallResCNNv5(num_classes=num_classes, base_channels=base_channels, dropout_p=dropout_p).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    early_stopping = EarlyStopping(patience=5, min_delta=0.01)

    # --- Dataset sizes ---
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_ds)}")
    print(f"  Validation: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")

    # --- Training ---
    print("\nStarting training with best hyperparameters...")
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        early_stopping,
        num_epochs=FT_EPOCHS,
        device=device,
        save_path=f"best_model_experiment_{experiment}.pth"
    )

    # Save trained model
    torch.save(model.state_dict(), f"{MODELS_PATH}/best_model_experiment_{experiment}.pth")

    # --- Evaluation ---
    test_loss, test_acc, test_f1, all_labels, all_preds = test_model(model, test_loader, criterion, device=device)
    print(f"\nTest Results - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")

    model_path = Path(f"{MODELS_PATH}/best_model_experiment_{modelstr}_{target_class}_{modelstr_save_name}_exp_{experiment}.pth")
    if model_path.parent.exists():
        experiment_id = str(model_path).split('_')[-1].split('.')[0]
        # experiment = int(experiment_id) + 1
        model_path = Path(f"{MODELS_PATH}/best_model_experiment_{modelstr}_{target_class}_{modelstr_save_name}_exp_{experiment}.pth")

    torch.save(model.state_dict(), model_path)

    # --- Save outputs ---
    pd.DataFrame({'labels': all_labels, 'preds': all_preds}).to_csv(
        f"{RESULTS_PATH}/test_scores_{modelstr}_{target_class}_{modelstr_save_name}_exp_{experiment}.csv", index=False
    )

    pd.DataFrame(history).to_csv(
        f"{RESULTS_PATH}/history_{modelstr}_{target_class}_{modelstr_save_name}_exp_{experiment}.csv", index=False
    )

    pd.DataFrame({
        'test_loss': [test_loss],
        'test_acc': [test_acc],
        'test_f1': [test_f1]
    }).to_csv(f"{RESULTS_PATH}/test_results_supervised_{modelstr}_{target_class}_{modelstr_save_name}_exp_{experiment}.csv", index=False)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs',fontsize=14)
    plt.ylabel('Loss',fontsize=14)
    plt.title('Training and Validation Loss',fontsize=14)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs',fontsize=14)
    plt.ylabel('Top-1 Accuracy',fontsize=14)
    plt.title('Training and Validation Accuracy',fontsize=14)
    plt.legend()
    plt.tight_layout()
    # plt.grid()
    plt.savefig(f'./results/figures/training_validation_history_{DATA_SENTINEL}.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()