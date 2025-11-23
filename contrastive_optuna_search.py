import click
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
from models.cnn import ContrastiveCNN
from models.vit import ContrastiveViT
from dataset import AugSpectrogramDataset, Augmentation
from config import EXPORT_DATA_PATH, RESULTS_PATH, SEED, SAMPLING_RATE, DATA_SENTINEL,EPOCHS
from models.losses import BatchAllTtripletLoss, SupConLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

@click.command()
@click.option('--experiment', default=100, type=int)
@click.option('--target_class', default=DATA_SENTINEL, type=str)
@click.option('--contrastive_method', default='triplet', type=click.Choice(['triplet','supcon']))
@click.option('--n_trials', default=2, type=int)
@click.option('--model_name', default='CustomCNNModel', help='Model architecture name (e.g. CustomCNNModel, ViTModel)')
@click.option('--modelstr', default='resnet18', help='Model architecture to use if CustomCNNModel is selected')
def main(experiment, target_class, contrastive_method, n_trials, model_name, modelstr):

    def objective(trial):
        # --- Hyperparameters ---
        lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8,16,32])
        duration = trial.suggest_categorical("duration", [1.0,2.0])
        latent_dim = trial.suggest_categorical("latent_dim", [64, 128, 256, 512])

        # --- Augmentation ---
        time_stretch_rate = trial.suggest_float("time_stretch_rate", 0.9, 1.1)
        pitch_shift_semitones = trial.suggest_int("pitch_shift_semitones", -2, 2)
        shift_prob = trial.suggest_float("shift_prob", 0.2, 0.5)
        augment = Augmentation(
            time_stretch=(time_stretch_rate, time_stretch_rate, 0.3),
            pitch_shift=(pitch_shift_semitones, pitch_shift_semitones, 0.3),
            shift_p=shift_prob
        )

        # --- Datasets ---
        train_dataset = AugSpectrogramDataset(
            f"{EXPORT_DATA_PATH}/train",
            duration=duration,
            target_sample_rate=SAMPLING_RATE,
            transform=augment
        )
        val_dataset = AugSpectrogramDataset(
            f"{EXPORT_DATA_PATH}/val",
            duration=duration,
            target_sample_rate=SAMPLING_RATE
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        # --- Model & Loss ---
        if model_name == "CustomCNNModel":
            model = ContrastiveCNN(latent_dim=latent_dim, weights=None, modelstr=modelstr).to(device)
        else:
            model = ContrastiveViT(model_name='vit_base_patch16_224', latent_dim=latent_dim, pretrained=False, in_chans=1).to(device)

        criterion = BatchAllTtripletLoss() if contrastive_method=='triplet' else SupConLoss(temperature=0.07)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # --- Training loop (short for search) ---
        model.train()
        for epoch in range(EPOCHS):
            for inputs, labels in train_loader:
                x = inputs['data'].to(device)
                y = labels.to(device)
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

        # --- Validation loss ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                x = inputs['data'].to(device)
                y = labels.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        return val_loss  # minimize contrastive loss

    # --- Optuna study ---
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # --- Save best hyperparameters ---
    trial = study.best_trial
    if model_name == "CustomCNNModel":
        modelstr_save_name = modelstr
    else:
        modelstr_save_name = "ViTModel"
    best_params_path = f'{RESULTS_PATH}/best_contrastive_hyperparams_{modelstr}_{target_class}_{contrastive_method}_{modelstr_save_name}_exp_{experiment}.json'
    with open(best_params_path, 'w') as f:
        json.dump(trial.params, f, indent=4)

    print(f"Best hyperparameters saved to {best_params_path}")
    print("Best trial validation loss:", trial.value)
    print("Best hyperparameters:", trial.params)

if __name__=="__main__":
    main()