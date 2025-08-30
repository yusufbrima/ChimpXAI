import torch
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, accuracy_score
from torch.utils.data import DataLoader

from dataset import SpectrogramDataset
from config import (
    RESULTS_PATH,
    SAMPLING_RATE,
    CHIMPANZEE_DATA_PATH
)

def extract_spectrogram_features(dataset: SpectrogramDataset):
    """
    Extract flattened features and labels from SpectrogramDataset
    """
    features, labels = [], []
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for sample, label in loader:
        spec = sample['data'].squeeze().numpy()
        features.append(spec.flatten())
        labels.append(label.item())
    return np.array(features), np.array(labels)


def train_model_on_spectrograms(
    model: ClassifierMixin,
    dataset_path: str,
    val_dataset_path: str = None,
    duration: int = 2,
    target_sample_rate: int = SAMPLING_RATE,
    random_state: int = 42,
    average: str = 'weighted',
) -> dict:
    """
    Train a classifier on spectrogram features and return metrics
    """
    # Prepare datasets
    train_ds = SpectrogramDataset(dataset_path, duration=duration, target_sample_rate=target_sample_rate)
    if val_dataset_path:
        test_ds = SpectrogramDataset(val_dataset_path, duration=duration, target_sample_rate=target_sample_rate)
        X_train, y_train = extract_spectrogram_features(train_ds)
        X_test, y_test = extract_spectrogram_features(test_ds)
    else:
        X, y = extract_spectrogram_features(train_ds)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )

    # Fit model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=average)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=train_ds.classes, zero_division=0))
    print(f"Accuracy: {acc:.4f} | {average.capitalize()} F1 Score: {f1:.4f}\n")

    return {
        'model': model,
        'accuracy': acc,
        'f1_score': f1,
        'classes': train_ds.classes,
    }


if __name__ == "__main__":
    NUM_RUNS = 50  # Number of times to repeat training
    MODELS = {
        'RandomForest': RandomForestClassifier(n_estimators=100),
        'XGBoost': xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=len(SpectrogramDataset(f"{CHIMPANZEE_DATA_PATH}/train", duration=2, target_sample_rate=SAMPLING_RATE).classes),
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100,
        ),
    }

    # Collect results
    history = []
    for run in range(1, NUM_RUNS + 1):
        print(f"\n=== Run {run}/{NUM_RUNS} ===")
        for name, model in MODELS.items():
            print(f"Training {name}...")
            # Re-instantiate simple models to reset state
            if name == 'RandomForest':
                clf = RandomForestClassifier(n_estimators=100)
            else:
                clf = xgb.XGBClassifier(
                    objective='multi:softprob',
                    num_class=len(SpectrogramDataset(
                        f"{CHIMPANZEE_DATA_PATH}/train", duration=2, target_sample_rate=SAMPLING_RATE
                    ).classes),
                    max_depth=5,
                    learning_rate=0.1,
                    n_estimators=100,
                )

            results = train_model_on_spectrograms(
                clf,
                f"{CHIMPANZEE_DATA_PATH}/train",
                f"{CHIMPANZEE_DATA_PATH}/val",
                random_state=run,
            )
            history.append({
                'run': run,
                'model': name,
                'accuracy': results['accuracy'],
                'f1_score': results['f1_score'],
            })

    # Create DataFrame and compute summary statistics
    df = pd.DataFrame(history)
    summary = df.groupby('model').agg(
        accuracy_mean=('accuracy', 'mean'),
        accuracy_std=('accuracy', 'std'),
        f1_mean=('f1_score', 'mean'),
        f1_std=('f1_score', 'std'),
    ).reset_index()

    print("\n=== Summary Across Runs ===")
    print(summary)

    # Save to CSV
    df.to_csv(f"{RESULTS_PATH}/all_model_runs.csv", index=False)
    summary.to_csv(f"{RESULTS_PATH}/summary_stats.csv", index=False)
