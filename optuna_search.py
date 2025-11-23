from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset import AugSpectrogramDataset, Augmentation
from models.cnn import SmallResCNNv5, CustomCNNModel  # import all models you want to support
from models.vit import ViTModel
from config import EXPORT_DATA_PATH, SAMPLING_RATE, RESULTS_PATH, SEED, DATA_SENTINEL,EPOCHS
from utils import train_model, test_model, EarlyStopping, compute_class_weights
from sklearn.metrics import f1_score
import click
import optuna
import json

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seeds
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


@click.command()
@click.option('--experiment', default=103, type=int, help='Experiment number')
@click.option('--target_class', default=DATA_SENTINEL, help='Target class for classification')
@click.option('--n_trials', default=2, type=int, help='Number of Optuna trials')
@click.option('--model_name', default='CustomCNNModel', help='Model architecture name (e.g. SmallResCNNv5, CustomCNNModel, ViTModel)')
@click.option('--modelstr', default='resnet18', help='Model architecture to use if CustomCNNModel is selected')
def main(experiment, target_class, n_trials, model_name, modelstr):

    def objective(trial):
        # --- Hyperparameters ---
        base_channels = trial.suggest_categorical("base_channels", [64, 96, 128, 192, 256])
        dropout_p = trial.suggest_float("dropout_p", 0.2, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        duration = trial.suggest_categorical("duration", [1.0, 2.0])

        # --- Augmentation ---
        time_stretch_rate = trial.suggest_float("time_stretch_rate", 0.9, 1.1)
        pitch_shift_semitones = trial.suggest_int("pitch_shift_semitones", -2, 2)
        shift_prob = trial.suggest_float("shift_prob", 0.2, 0.5)

        augment = Augmentation(
            time_stretch=(time_stretch_rate, time_stretch_rate, 0.3),
            pitch_shift=(pitch_shift_semitones, pitch_shift_semitones, 0.3),
            shift_p=shift_prob
        )

        # --- Spectrogram parameters ---
        n_fft = trial.suggest_categorical("n_fft", [256, 512, 1024])
        hop_length = trial.suggest_categorical("hop_length", [128, 256, 512])

        # --- Datasets ---
        train_dataset = AugSpectrogramDataset(
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

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        # --- Class weights ---
        train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
        class_weights_eff = compute_class_weights(train_labels, method='effective', beta=0.99, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_eff)

        # --- Model ---
        num_classes = len(train_dataset.classes)
        if model_name == "CustomCNNModel":
            model = CustomCNNModel(num_classes=num_classes, weights=None, modelstr=modelstr).to(device)
        if model_name == "ViTModel":
            model =ViTModel(model_name='vit_base_patch16_224', num_classes=num_classes, pretrained=False, in_chans=1).to(device)
        else:
            model = SmallResCNNv5(num_classes=num_classes, base_channels=base_channels, dropout_p=dropout_p).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        early_stopping = EarlyStopping(patience=5, min_delta=0.01)

        # --- Train for few epochs during search ---
        _ = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                        early_stopping=early_stopping, num_epochs=EPOCHS, device=device)

        # --- Validation evaluation ---
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for samples, labels in val_loader:
                x, y = samples["data"].to(device), labels.to(device)
                outputs = model(x)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_f1 = f1_score(all_labels, all_preds, average='macro')
        return val_f1

    # --- Run Optuna study ---
    study = optuna.create_study(direction="maximize", study_name="Chimpanzee_Hyperparam_Optimization")
    study.optimize(objective, n_trials=n_trials)

    # --- Best trial ---
    trial = study.best_trial
    print("Best trial F1:", trial.value)
    print("Best hyperparameters:", trial.params)

    # --- Save best hyperparameters ---
    if model_name == "CustomCNNModel":
        modelstr_save_name = modelstr
    elif model_name == "ViTModel":
        modelstr_save_name = "ViTModel"
    else:
        modelstr_save_name = "SmallResCNNv5"
    #  'results/metrics/best_hyperparams_experiment_dense121_100_chimp.json'
    best_params_path = f'{RESULTS_PATH}/best_hyperparams_experiment_{modelstr_save_name}_{experiment}_{target_class}.json'
    with open(best_params_path, 'w') as f:
        json.dump(trial.params, f, indent=4)
    print(f"Best hyperparameters saved to {best_params_path}")

    # --- Train final model with best hyperparameters ---
    best_params = trial.params
    augment = Augmentation(
        time_stretch=(best_params["time_stretch_rate"], best_params["time_stretch_rate"], 0.3),
        pitch_shift=(best_params["pitch_shift_semitones"], best_params["pitch_shift_semitones"], 0.3),
        shift_p=best_params["shift_prob"]
    )

    train_dataset = AugSpectrogramDataset(
        f"{EXPORT_DATA_PATH}/train",
        duration=best_params["duration"],
        target_sample_rate=SAMPLING_RATE,
        transform=augment,
        n_fft=best_params["n_fft"],
        hop_length=best_params["hop_length"]
    )
    val_dataset = AugSpectrogramDataset(
        f"{EXPORT_DATA_PATH}/val",
        duration=best_params["duration"],
        target_sample_rate=SAMPLING_RATE,
        n_fft=best_params["n_fft"],
        hop_length=best_params["hop_length"]
    )
    test_ds = AugSpectrogramDataset(
        f"{EXPORT_DATA_PATH}/test",
        duration=best_params["duration"],
        target_sample_rate=SAMPLING_RATE,
        n_fft=best_params["n_fft"],
        hop_length=best_params["hop_length"]
    )

    train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=best_params["batch_size"], shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=best_params["batch_size"], shuffle=False, num_workers=2)

    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    class_weights_eff = compute_class_weights(train_labels, method='effective', beta=0.99, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_eff)

    # --- Final model ---
    num_classes = len(train_dataset.classes)
    if model_name == "CustomCNNModel":
        model = CustomCNNModel(num_classes=num_classes, weights=None, modelstr=modelstr).to(device)
    elif model_name == "ViTModel":
        model =ViTModel(model_name='vit_base_patch16_224', num_classes=num_classes, pretrained=False, in_chans=1).to(device)
    else:
        model = SmallResCNNv5(num_classes=num_classes, base_channels=best_params["base_channels"], dropout_p=best_params["dropout_p"]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    early_stopping = EarlyStopping(patience=5, min_delta=0.01)

    # --- Train final model ---
    model, history = train_model(model, train_loader, val_loader, criterion, optimizer,
                                 scheduler=scheduler, early_stopping=early_stopping, num_epochs=EPOCHS, device=device,
                                 save_path=f'best_model_experiment_{experiment}.pth')

    # --- Evaluate test set ---
    test_loss, test_acc, test_f1, all_labels, all_preds = test_model(model, test_loader, criterion, device=device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")

    # --- Save results ---
    pd.DataFrame({'labels': all_labels, 'preds': all_preds}).to_csv(f'{RESULTS_PATH}/test_scores_experiment_{modelstr_save_name}_{experiment}_{target_class}.csv', index=False)
    pd.DataFrame(history).to_csv(f'{RESULTS_PATH}/history_experiment_{modelstr_save_name}_{experiment}_{target_class}.csv', index=False)
    pd.DataFrame({'test_loss': [test_loss], 'test_acc': [test_acc], 'test_f1': [test_f1]}).to_csv(f'{RESULTS_PATH}/test_results_experiment_{modelstr_save_name}_{experiment}_{target_class}.csv', index=False)


if __name__ == "__main__":
    main()
