import click
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

from models.cnn import ContrastiveCNN, FinetuningClassifier
from models.vit import ContrastiveViT,  ViTFinetuningClassifier
from dataset import AugSpectrogramDataset, Augmentation
from config import EXPORT_DATA_PATH, RESULTS_PATH, SEED, SAMPLING_RATE,FT_EPOCHS,DATA_SENTINEL,MODELS_PATH
from utils import EarlyStopping, train_model, test_model, compute_class_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

import uuid
hash_str = uuid.uuid4().hex

@click.command()
@click.option('--experiment', default=100, type=int)
@click.option('--target_class', default=DATA_SENTINEL, type=str)
@click.option('--contrastive_method', default='triplet', type=click.Choice(['triplet','supcon']))
@click.option('--model_name', default='CustomCNNModel', type=str, help='Model architecture: CustomCNNModel, ViTModel')
@click.option('--modelstr', default='resnet18', type=str, help='Model type for CustomCNNModel')
@click.option('--num_epochs', default=FT_EPOCHS, type=int)
def main(experiment, target_class, contrastive_method, model_name, modelstr, num_epochs):

    # ----------------------
    # Determine hyperparameter JSON path
    # ----------------------
    # modelstr_save_name = modelstr if model_name == "CustomCNNModel" else model_name
    # best_params_path = Path(f'{RESULTS_PATH}/best_hyperparams_experiment_{modelstr_save_name}_{experiment}_{target_class}.json')
    # best_params_path = Path(f'{RESULTS_PATH}/best_contrastive_hyperparams_{modelstr}_{target_class}_{contrastive_method}_{modelstr_save_name}_exp_{experiment}.json')
    if model_name == "CustomCNNModel":
        modelstr_save_name = modelstr
    else:
        modelstr_save_name = "ViTModel"
    
    best_params_path = Path(f'{RESULTS_PATH}/best_contrastive_hyperparams_{modelstr}_{target_class}_{contrastive_method}_{modelstr_save_name}_exp_{experiment}.json')

    if not best_params_path.exists():
        raise FileNotFoundError(f"Best hyperparameters not found at {best_params_path}")

    with open(best_params_path, 'r') as f:
        best_params = json.load(f)

    batch_size = best_params["batch_size"]
    duration = best_params["duration"]
    latent_dim = best_params.get("latent_dim", 128)
    n_fft = best_params.get("n_fft", 512)
    hop_length = best_params.get("hop_length", 256)
    lr = best_params.get("lr", 1e-4)

    # ----------------------
    # Augmentation
    # ----------------------
    augment = Augmentation(
        time_stretch=(best_params.get("time_stretch_rate",1.0), best_params.get("time_stretch_rate",1.0), 0.3),
        pitch_shift=(best_params.get("pitch_shift_semitones",0), best_params.get("pitch_shift_semitones",0), 0.3),
        shift_p=best_params.get("shift_prob",0.3)
    )

    # ----------------------
    # Datasets & Loaders
    # ----------------------
    train_dataset = AugSpectrogramDataset(f"{EXPORT_DATA_PATH}/train", duration=duration, target_sample_rate=SAMPLING_RATE,
                                          transform=augment, n_fft=n_fft, hop_length=hop_length)
    val_dataset = AugSpectrogramDataset(f"{EXPORT_DATA_PATH}/val", duration=duration, target_sample_rate=SAMPLING_RATE,
                                        n_fft=n_fft, hop_length=hop_length)
    test_dataset = AugSpectrogramDataset(f"{EXPORT_DATA_PATH}/test", duration=duration, target_sample_rate=SAMPLING_RATE,
                                         n_fft=n_fft, hop_length=hop_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # ----------------------
    # Class weights
    # ----------------------
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    class_weights = compute_class_weights(train_labels, method='effective', beta=0.99, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ----------------------
    # Load contrastive-pretrained model
    # ----------------------
    if model_name == "CustomCNNModel":
        contrastive_model = ContrastiveCNN(latent_dim=latent_dim, modelstr=modelstr).to(device)
    elif model_name == "ViTModel":
        contrastive_model = ContrastiveViT(model_name='vit_base_patch16_224', latent_dim=latent_dim,
                                           pretrained=False, in_chans=1).to(device)
    else:
        contrastive_model = ContrastiveCNN(latent_dim=latent_dim, modelstr="SmallResCNNv5").to(device)

    # pretrained_path = Path(f"{RESULTS_PATH}/contrastive_trained_model_{modelstr}_{target_class}_{contrastive_method}_exp_{experiment}.pt")
    # pretrained_path = Path(f"{MODELS_PATH}/contrastive_trained_model_{modelstr}_{target_class}_{contrastive_method}_{modelstr_save_name}_exp_{experiment}.pt")
    pretrained_path = Path(f"{MODELS_PATH}/contrastive_trained_model_{modelstr}_{target_class}_{contrastive_method}_{modelstr_save_name}_exp_{experiment}.pt")

    if not pretrained_path.exists():
        raise FileNotFoundError(f"Pretrained contrastive model not found at {pretrained_path}")

    contrastive_model.load_state_dict(torch.load(pretrained_path, map_location=device))

    # ----------------------
    # Fine-tuning classifier
    # ----------------------
    if model_name == "ViTModel":
        classifier = ViTFinetuningClassifier(contrastive_model, num_classes=len(train_dataset.classes), requires_grad=True).to(device)
    else:
        classifier = FinetuningClassifier(contrastive_model, num_classes=len(train_dataset.classes), requires_grad=True).to(device)

    optimizer = optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    early_stopping = EarlyStopping(patience=10, min_delta=0.01)

    # ----------------------
    # Train classifier
    # ----------------------
    classifier, history = train_model(
        classifier,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping,
        num_epochs=num_epochs,
        device=device,
        save_path=f"finetuned_model_{modelstr_save_name}_{target_class}_{hash_str}_{experiment}.pth"
    )

    # ----------------------
    # Evaluate on test set
    # ----------------------
    test_loss, test_acc, test_f1, all_labels, all_preds = test_model(classifier, test_loader, criterion, device=device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")
    model_path = Path(f"{MODELS_PATH}/finetuned_model_{modelstr}_{target_class}_{contrastive_method}_{modelstr_save_name}_exp_{experiment}.pth")
    torch.save(classifier.state_dict(), model_path)


    # ----------------------
    # Save results
    # ----------------------
    
    pd.DataFrame({'labels': all_labels, 'preds': all_preds}).to_csv(
        f'{RESULTS_PATH}/test_predictions_{modelstr}_{target_class}_{modelstr_save_name}_exp_{experiment}.csv', index=False)
    pd.DataFrame(history).to_csv(
        f'{RESULTS_PATH}/history_{modelstr}_{target_class}_{modelstr_save_name}_exp_{experiment}.csv', index=False)
    pd.DataFrame({'test_loss':[test_loss],'test_acc':[test_acc],'test_f1':[test_f1]}).to_csv(
        f'{RESULTS_PATH}/test_results_contrastive_{modelstr}_{target_class}_{modelstr_save_name}_{contrastive_method}_exp_{experiment}.csv', index=False)


if __name__=="__main__":
    main()
