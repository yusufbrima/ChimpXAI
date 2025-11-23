import click
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from dataset import AugSpectrogramDataset, Augmentation
from config import EXPORT_DATA_PATH, RESULTS_PATH, SEED, SAMPLING_RATE,DATA_SENTINEL,EPOCHS,MODELS_PATH
from models.cnn import ContrastiveCNN
from models.vit import ContrastiveViT
from models.losses import BatchAllTtripletLoss, SupConLoss
from pathlib import Path
# ----------------------
# Device
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

@click.command()
@click.option('--experiment', default=100, type=int, help='Experiment number')
@click.option('--target_class', default=DATA_SENTINEL, type=str, help='Target class')
@click.option('--contrastive_method', default='triplet', type=click.Choice(['triplet','supcon']), help='Contrastive loss method')
@click.option('--model_name', default='CustomCNNModel', type=str, help='Model type: CustomCNNModel or ViTModel')
@click.option('--modelstr', default='resnet18', type=str, help='Model string (for CNN backbones)')
@click.option('--num_epochs', default=EPOCHS, type=int, help='Number of training epochs')
def main(experiment, target_class, contrastive_method, model_name, modelstr, num_epochs):

    # ----------------------
    # Load best hyperparameters
    # ----------------------
    # hyperparams_file = f"{RESULTS_PATH}/best_contrastive_hyperparams_{modelstr}_{target_class}_{contrastive_method}_{'ViTModel' if model_name!='CustomCNNModel' else modelstr}_exp_{experiment}.json"

    # best_contrastive_hyperparams_resnet18_chimp_supcon_resnet18_exp_100.json

    if model_name == "CustomCNNModel":
        modelstr_save_name = modelstr
    else:
        modelstr_save_name = "ViTModel"
    
    hyperparams_file = f'{RESULTS_PATH}/best_contrastive_hyperparams_{modelstr}_{target_class}_{contrastive_method}_{modelstr_save_name}_exp_{experiment}.json'

    with open(hyperparams_file, 'r') as f:
        best_params = json.load(f)
    print("Loaded hyperparameters:", best_params)

    # ----------------------
    # Augmentation
    # ----------------------
    augment = Augmentation(
        time_stretch=(best_params["time_stretch_rate"], best_params["time_stretch_rate"], 0.3),
        pitch_shift=(best_params["pitch_shift_semitones"], best_params["pitch_shift_semitones"], 0.3),
        shift_p=best_params["shift_prob"]
    )

    # ----------------------
    # Datasets & Loaders
    # ----------------------
    train_dataset = AugSpectrogramDataset(
        f"{EXPORT_DATA_PATH}/train",
        duration=best_params["duration"],
        target_sample_rate=SAMPLING_RATE,
        transform=augment
    )
    val_dataset = AugSpectrogramDataset(
        f"{EXPORT_DATA_PATH}/val",
        duration=best_params["duration"],
        target_sample_rate=SAMPLING_RATE
    )

    train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=best_params["batch_size"], shuffle=False, num_workers=2)

    # ----------------------
    # Model & Loss
    # ----------------------
    if model_name == "CustomCNNModel":
        model = ContrastiveCNN(latent_dim=best_params["latent_dim"], weights=None, modelstr=modelstr).to(device)
    else:
        model = ContrastiveViT(model_name='vit_base_patch16_224', latent_dim=best_params["latent_dim"], pretrained=False, in_chans=1).to(device)

    criterion = BatchAllTtripletLoss() if contrastive_method=='triplet' else SupConLoss(temperature=0.07)
    optimizer = optim.Adam(model.parameters(), lr=best_params["lr"])

    # ----------------------
    # Training loop
    # ----------------------
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for inputs, labels in train_loader:
            x = inputs['data'].to(device)
            y = labels.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        train_loss = running_train_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation loss
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                x = inputs['data'].to(device)
                y = labels.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                running_val_loss += loss.item()
        val_loss = running_val_loss / len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    # ----------------------
    # Save model and losses
    # ----------------------
    model_save_path = f"{MODELS_PATH}/contrastive_trained_model_{modelstr}_{target_class}_{contrastive_method}_{modelstr_save_name}_exp_{experiment}.pt"
    if Path(model_save_path).parent.exists():
        experiment_id = str(model_save_path).split('_')[-1].split('.')[0]
        experiment = int(experiment_id) + 1
        model_save_path = f"{MODELS_PATH}/contrastive_trained_model_{modelstr}_{target_class}_{contrastive_method}_{modelstr_save_name}_exp_{experiment}.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    losses_df = pd.DataFrame({"train_loss": train_losses, "val_loss": val_losses})
    losses_df.to_csv(f"{RESULTS_PATH}/contrastive_training_losses_{modelstr}_{target_class}_{contrastive_method}_{modelstr_save_name}_exp_{experiment}.csv", index=False)
    print("Training losses saved.")

if __name__ == "__main__":
    main()
