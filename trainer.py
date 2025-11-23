from pathlib import Path
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
import torchaudio.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np
from torch.utils.data import DataLoader
from data.dataset import AugSpectrogramDataset,Augmentation
from models.cnn import CustomCNNModel,SmallResCNNv5
from config import CLASSIFIER_BATCH_SIZE, MODELS_PATH, RESULTS_PATH,SAMPLING_RATE,FT_EPOCHS,EXPORT_DATA_PATH,DATA_SENTINEL
from utils import train_model, test_model, EarlyStopping, plot_confusion_matrix,compute_class_weights
from tqdm import tqdm  # Add tqdm for progress bar
import click
from collections import Counter
import uuid
hash_str = uuid.uuid4().hex

# Set random seed for reproducibility
# torch.manual_seed(SEED)
# np.random.seed(SEED)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@click.command()
@click.option('--modelstr', default='resnet18', help='Model architecture to use')
@click.option('--experiment', default=52, type=int, help='Experiment number')
@click.option('--target_class', default=DATA_SENTINEL, help='Target class for classification')
@click.option('--weighting', default='effective', help='Class weighting method: inverse or effective')
def main(modelstr, experiment,target_class, weighting):
    
    augment = Augmentation()

    # Load the dataset
    train_dataset = AugSpectrogramDataset(f"{EXPORT_DATA_PATH}/train", duration=2, target_sample_rate=SAMPLING_RATE,transform=augment)
    val_dataset = AugSpectrogramDataset(f"{EXPORT_DATA_PATH}/val", duration=2, target_sample_rate=SAMPLING_RATE)

    test_dataset = AugSpectrogramDataset(f"{EXPORT_DATA_PATH}/test", duration=2, target_sample_rate=SAMPLING_RATE)

    # Define the data loaders
    train_loader = DataLoader(train_dataset, batch_size=CLASSIFIER_BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CLASSIFIER_BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=CLASSIFIER_BATCH_SIZE, shuffle=False, num_workers=2)

    # Extract labels from the training dataset
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]

    # Compute class weights based on the specified weighting method
    class_weights = compute_class_weights(train_labels, method=weighting, device=device) # effective, inverse, none




    sample, label = next(iter(train_loader))

    input_shape = sample['data'].shape[1:]  

    num_classes = len(train_dataset.classes)

    # print(f"Input shape: {input_shape}, Number of classes: {num_classes}")

    model = CustomCNNModel(num_classes=num_classes, weights=None, modelstr=modelstr)
    # model = SmallCNNModel(num_classes=num_classes, input_height=input_shape[1], input_width=input_shape[2])
    # model = SmallResCNNv2(num_classes=num_classes, dropout_p=0.3)
    # model = SmallResCNNv3(num_classes=num_classes, dropout_p=0.3)
    # model = SmallResCNNv5(num_classes=num_classes, dropout_p=0.3)
    model = model.to(device)

    # Define loss function and optimizer
    #weight=torch.tensor(CLASS_WEIGHTS).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00018667743960289472, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    early_stopping = EarlyStopping(patience=10, min_delta=0.01)

    # print the length of the datasets 
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # # Train the model
    model, history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, num_epochs=FT_EPOCHS, device=device, save_path=f'saved_model_{modelstr}_{target_class}_experiment_{experiment}_{hash_str}.pth')
    
    # # Save the trained model
    torch.save(model.state_dict(), f'{MODELS_PATH}/custom_{modelstr}_{target_class}_experiment_{experiment}.pth')

    # # Example usage:
    test_loss, test_acc, test_f1, all_labels, all_preds = test_model(model, test_loader, criterion, device=device)
    # test_loss, test_acc, all_labels, all_preds = test_model(model, test_loader, criterion, device=device)

    # # Save the test labels and predictions
    test_results = {'labels': all_labels, 'preds': all_preds}
    test_results_df = pd.DataFrame(test_results)
    test_results_df.to_csv(f'{RESULTS_PATH}/{modelstr}_test_scores_{target_class}_experiment_{experiment}.csv', index=False)

    # # Save the training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(f'{RESULTS_PATH}/{modelstr}_history_{target_class}_experiment_{experiment}.csv', index=False)

    # # Save the test results
    test_results = {'test_loss': test_loss, 'test_acc': test_acc, 'test_f1': test_f1}
    test_results_df = pd.DataFrame(test_results, index=[0])
    test_results_df.to_csv(f'{RESULTS_PATH}/{modelstr}_test_results_{target_class}_experiment_{experiment}.csv', index=False)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f} Test F1: {test_f1:.4f}')

if __name__=="__main__":
    main()
