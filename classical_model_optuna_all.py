"""
Hyperparameter Search for Classical ML Models
=============================================

This script performs Optuna-based hyperparameter optimization for:
- Decision Tree
- Random Forest
- XGBoost

Both model-specific and dataset-specific parameters are optimized.
The results are saved in JSON format, including F1 scores and best parameters.

Usage:
    python search_classical_models.py --experiment 501 --n_trials 20 --seed 42
"""

import argparse
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from data.dataset import AugSpectrogramDataset, Augmentation
from config import EXPORT_DATA_PATH, SAMPLING_RATE, RESULTS_PATH, SEED, DATA_SENTINEL


# ======================================================
# Utilities
# ======================================================
def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    print(f"[INFO] Random seed set to {seed}")


def extract_features_from_loader(loader):
    """Flatten spectrogram tensors into 1D features for classical ML."""
    X, y = [], []
    for samples, labels in tqdm(loader, desc="Extracting features"):
        spec = samples["data"].numpy()
        b, c, h, w = spec.shape
        X.append(spec.reshape(b, -1))
        y.extend(labels.numpy())
    return np.concatenate(X, axis=0), np.array(y)


def create_datasets(trial):
    """Create train/val dataloaders with trial-specific augmentation & spectrogram parameters."""
    # --- Search space for dataset ---
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    duration = trial.suggest_categorical("duration", [1.0, 2.0])
    time_stretch_rate = trial.suggest_float("time_stretch_rate", 0.9, 1.1)
    pitch_shift_semitones = trial.suggest_int("pitch_shift_semitones", -2, 2)
    shift_prob = trial.suggest_float("shift_prob", 0.2, 0.5)
    n_fft = trial.suggest_categorical("n_fft", [256, 512, 1024])
    hop_length = trial.suggest_categorical("hop_length", [128, 256, 512])

    augment = Augmentation(
        time_stretch=(time_stretch_rate, time_stretch_rate, 0.3),
        pitch_shift=(pitch_shift_semitones, pitch_shift_semitones, 0.3),
        shift_p=shift_prob
    )

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

    dataset_params = {
        "batch_size": batch_size,
        "duration": duration,
        "time_stretch_rate": time_stretch_rate,
        "pitch_shift_semitones": pitch_shift_semitones,
        "shift_prob": shift_prob,
        "n_fft": n_fft,
        "hop_length": hop_length
    }

    return train_loader, val_loader, dataset_params


def get_model_and_params(method, trial):
    """Define each classical model’s search space."""
    if method == "decision_tree":
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 32),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "criterion": trial.suggest_categorical("criterion",['gini', 'entropy', 'log_loss']),
        }
        model = DecisionTreeClassifier(**params, random_state=SEED)

    elif method == "random_forest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 4, 32),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "criterion": trial.suggest_categorical("criterion",['gini', 'entropy', 'log_loss']),
        }
        model = RandomForestClassifier(**params, n_jobs=-1, random_state=SEED)

    elif method == "xgboost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        # tree_method="gpu_hist",predictor="gpu_predictor"
        model = XGBClassifier(**params, eval_metric="mlogloss",tree_method = "hist", device = "cuda", max_bin=64, random_state=SEED)

    else:
        raise ValueError(f"Unknown method: {method}")

    return model, params


def run_all_methods(n_trials, seed):
    """Run optimization for all classical models sequentially and store results."""
    results = {}
    sampler = optuna.samplers.TPESampler(seed=seed)
    # ["decision_tree", "random_forest", "xgboost"]
    for method in ["decision_tree", "random_forest", "xgboost"]:
        print(f"\n=== Optimization for {method.upper()} ===")

        def objective(trial):
            train_loader, val_loader, dataset_params = create_datasets(trial)
            X_train, y_train = extract_features_from_loader(train_loader)
            X_val, y_val = extract_features_from_loader(val_loader)

            model, model_params = get_model_and_params(method, trial)
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", model)
            ])

            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_val)
            return f1_score(y_val, preds, average="weighted")

        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials)

        best_trial = study.best_trial
        results[method] = {
            "best_f1": best_trial.value,
            "params": best_trial.params,  # includes model + dataset search params
        }

    return results


def main(experiment=501, n_trials=2, seed=42):
    set_seed(seed)
    print("=== Classical ML Hyperparameter Search ===")

    results = run_all_methods(n_trials, seed)

    out_path = Path(f"{RESULTS_PATH}/classical_ml_all_best_{DATA_SENTINEL}_{experiment}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n All results saved to {out_path}")
    for method, res in results.items():
        print(f"{method}: F1={res['best_f1']:.4f}")


# ======================================================
# Entry Point
# ======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter search for classical ML models")
    parser.add_argument("--experiment", type=int, default=501, help="Experiment ID")
    parser.add_argument("--n_trials", type=int, default=2, help="Number of trials per model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    main(experiment=args.experiment, n_trials=args.n_trials, seed=args.seed)
