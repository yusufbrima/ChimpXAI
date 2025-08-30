import torch
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import numpy as np
# save the results for each model as a csv file
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report,accuracy_score
from torch.utils.data import DataLoader
from dataset import SpectrogramDataset
from config import DATA_PATH, CLASSIFIER_BATCH_SIZE, LEARNING_RATE, SEED, MODELS_PATH, RESULTS_PATH,SAMPLING_RATE,FT_EPOCHS,CHIMPANZEE_DATA_PATH,CLASS_WEIGHTS


def extract_spectrogram_features(dataset):
    """
    Extract flattened features and labels from SpectrogramDataset
    
    Parameters:
    -----------
    dataset : SpectrogramDataset
        The spectrogram dataset to extract features from
    
    Returns:
    --------
    tuple: (features, labels)
        features: numpy array of flattened spectrogram features
        labels: numpy array of class labels
    """
    features = []
    labels = []
    
    # Create a DataLoader to iterate through the dataset
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for sample, label in dataloader:
        # Extract log spectrogram from the sample
        spectrogram = sample['data'].squeeze().numpy()
        
        # Flatten the spectrogram
        flattened_features = spectrogram.flatten()
        
        features.append(flattened_features)
        labels.append(label.item())
    
    return np.array(features), np.array(labels)


def train_model_on_spectrograms(
    model, 
    dataset_path, 
    val_dataset_path=None, 
    duration=2, 
    target_sample_rate=SAMPLING_RATE, 
    random_state=42, 
    average='weighted'
):
    """
    Train a machine learning classifier on spectrogram features
    
    Parameters:
    -----------
    model : ClassifierMixin
        A scikit-learn compatible machine learning model
    dataset_path : str
        Path to the training dataset
    val_dataset_path : str, optional
        Path to the validation dataset. If None, will use a train-test split
    duration : int, optional
        Duration of audio clips
    target_sample_rate : int, optional
        Target sampling rate for audio processing
    random_state : int, optional
        Controls the shuffling applied to the data before split
    average : str, optional
        Method for calculating F1 score (macro, micro, weighted)
    
    Returns:
    --------
    dict: A dictionary containing model training results
    """
    # Create SpectrogramDataset for training
    train_ds = SpectrogramDataset(dataset_path, duration=duration, target_sample_rate=target_sample_rate)
    
    # Determine validation dataset
    if val_dataset_path:
        test_dataset = SpectrogramDataset(val_dataset_path, duration=duration, target_sample_rate=target_sample_rate)
        # Extract features for separate datasets
        X_train, y_train = extract_spectrogram_features(train_ds)
        X_test, y_test = extract_spectrogram_features(test_dataset)
    else:

        X, y = extract_spectrogram_features(train_ds)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    
    # Special handling for XGBoost if the model is an XGBoost model
    if isinstance(model, xgb.XGBClassifier):
        # For XGBoost classifier, use its native fit method
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        # For other scikit-learn compatible models
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=average)
    
    # Print detailed classification report
    print("Classification Report:")
    print(classification_report(
        y_test, 
        y_pred, 
        target_names=train_ds.classes, 
        zero_division=0
    ))
    
    # Print overall results
    print(f"\nNumber of classes: {len(train_ds.classes)}")
    print(f"Classes: {train_ds.classes}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"{average.capitalize()} F1 Score: {f1:.4f}")
    
    return {
        'model': model,
        'accuracy': accuracy,
        'f1_score': f1,
        'classes': train_ds.classes,
        'classification_report': classification_report(
            y_test, 
            y_pred, 
            target_names=train_ds.classes, 
            zero_division=0
        )
    }




if __name__=="__main__":
    train_ds = SpectrogramDataset(f"{CHIMPANZEE_DATA_PATH}/train", duration=2, target_sample_rate=SAMPLING_RATE)


    # Random Forest example
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_results = train_model_on_spectrograms(rf_model, f"{CHIMPANZEE_DATA_PATH}/train",f"{CHIMPANZEE_DATA_PATH}/val")

    print(rf_results['accuracy'])
    print(rf_results['f1_score'])
    print(rf_results['classification_report'])

    # XGBoost example
    xgb_model = xgb.XGBClassifier(
            objective='multi:softprob', 
            num_class=len(train_ds.classes),
            max_depth=5, 
            learning_rate=0.1, 
            n_estimators=100
        )
    xgb_results = train_model_on_spectrograms(xgb_model, f"{CHIMPANZEE_DATA_PATH}/train",f"{CHIMPANZEE_DATA_PATH}/val")


    print(xgb_results['accuracy'])
    print(xgb_results['f1_score'])
    print(xgb_results['classification_report'])


    result_history = {'model':[], 'accuracy':[], 'f1_score':[], 'classification_report':[]}
    for model_results in [rf_results, xgb_results]:
        result_history['model'].append(model_results['model'].__class__.__name__)
        result_history['accuracy'].append(model_results['accuracy'])
        result_history['f1_score'].append(model_results['f1_score'])
        result_history['classification_report'].append(model_results['classification_report'])
    result_history_df = pd.DataFrame(result_history)
    result_history_df.to_csv(f'{RESULTS_PATH}/model_results.csv', index=False)
