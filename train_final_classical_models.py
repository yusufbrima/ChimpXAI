"""
Evaluate Classical ML Models using Optimal Hyperparameters
==========================================================

Usage:
    python evaluate_classical_models.py --experiment 501 --seed 123

This script loads the best hyperparameters obtained from Optuna optimization
and retrains each model (Decision Tree, Random Forest, XGBoost, LightGBM,
CatBoost) on its own optimal dataset configuration. Evaluation is done on the test set.

Results (accuracy, F1, and predictions) are saved under RESULTS_PATH.
"""

import argparse
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from data.dataset import AugSpectrogramDataset, Augmentation
from config import EXPORT_DATA_PATH, SAMPLING_RATE, RESULTS_PATH, DATA_SENTINEL


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
    """Flatten spectrograms into tabular feature vectors."""
    X, y = [], []
    for samples, labels in tqdm(loader, desc="Extracting features"):
        spec = samples["data"].numpy()
        b, c, h, w = spec.shape
        spec_flat = spec.reshape(b, -1)
        X.append(spec_flat)
        y.extend(labels.numpy())
    X = np.concatenate(X, axis=0)
    y = np.array(y)
    return X, y


def create_datasets(best_params, duration=1.0, n_fft=512, hop_length=256):
    """Create train/val/test dataloaders using model-specific augmentation parameters."""
    augment = Augmentation(
        time_stretch=(best_params["time_stretch_rate"], best_params["time_stretch_rate"], 0.3),
        pitch_shift=(best_params["pitch_shift_semitones"], best_params["pitch_shift_semitones"], 0.3),
        shift_p=best_params["shift_prob"]
    )

    def make_loader(split):
        ds = AugSpectrogramDataset(
            f"{EXPORT_DATA_PATH}/{split}",
            duration=duration,
            target_sample_rate=SAMPLING_RATE,
            transform=augment if split == "train" else None,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        return DataLoader(ds, batch_size=best_params["batch_size"], shuffle=(split == "train"), num_workers=2)

    return make_loader("train"), make_loader("val"), make_loader("test")


def get_model(method, params, seed):
    """Recreate a model with the best hyperparameters."""
    if method == "decision_tree":
        return DecisionTreeClassifier(**params, random_state=seed)
    elif method == "random_forest":
        return RandomForestClassifier(**params, random_state=seed, n_jobs=-1)
    elif method == "xgboost":
        return XGBClassifier(**params, eval_metric="mlogloss",tree_method = "hist", device = "cuda", max_bin=64, random_state=seed)
    elif method == "lightgbm":
        return LGBMClassifier(**params, random_state=seed)
    elif method == "catboost":
        return CatBoostClassifier(**params, random_state=seed, verbose=0)
    else:
        raise ValueError(f"Unknown model: {method}")


# ======================================================
# Main
# ======================================================
def main(experiment: int, seed: int):
    """Load best hyperparams, train & evaluate final classical models."""
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    best_json_path = Path(f"{RESULTS_PATH}/classical_ml_all_best_{DATA_SENTINEL}_{experiment}.json")
    if not best_json_path.exists():
        raise FileNotFoundError(f"Missing: {best_json_path}")

    with open(best_json_path, "r") as f:
        best_params_all = json.load(f)

    results = {}

    # --- Train and Evaluate Each Model ---
    for method, info in best_params_all.items():
        print(f"\n=== Training Final {method.upper()} ===")
        best_params = info["params"]

        # Extract dataset-specific parameters
        duration = best_params["duration"]
        n_fft = best_params["n_fft"]
        hop_length = best_params["hop_length"]

        # --- Prepare dataset for this model ---
        print(f"[INFO] Preparing dataset for {method} (duration={duration}, n_fft={n_fft}, hop_length={hop_length})")
        train_loader, val_loader, test_loader = create_datasets(
            best_params, duration=duration, n_fft=n_fft, hop_length=hop_length
        )

        X_train, y_train = extract_features_from_loader(train_loader)
        X_val, y_val = extract_features_from_loader(val_loader)
        X_test, y_test = extract_features_from_loader(test_loader)

        X_trainval = np.concatenate([X_train, X_val])
        y_trainval = np.concatenate([y_train, y_val])

        # Extract classifier-specific parameters only
        clf_params = {
            k: v for k, v in best_params.items()
            if k not in [
                "batch_size", "duration", "n_fft", "hop_length",
                "time_stretch_rate", "pitch_shift_semitones", "shift_prob"
            ]
        }

        model = get_model(method, clf_params, seed)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", model)
        ])

        pipeline.fit(X_trainval, y_trainval)
        preds = pipeline.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        print(f"→ Test Accuracy: {acc:.4f} | F1: {f1:.4f}")
        print(classification_report(y_test, preds))

        results[method] = {
            "accuracy": acc,
            "f1": f1,
            "params": best_params
        }

        # Save per-model predictions
        pd.DataFrame({
            "y_true": y_test,
            "y_pred": preds
        }).to_csv(f"{RESULTS_PATH}/{method}_final_preds_{DATA_SENTINEL}_exp_{experiment}.csv", index=False)

    # Save summary

    summary_path = f"{RESULTS_PATH}/final_classical_results{DATA_SENTINEL}_exp_{experiment}.json"
    if Path(summary_path).exists():
        experiment_id = summary_path.split('_')[-1].split('.')[0]
        experiment = int(experiment_id) + 1
        summary_path = f"{RESULTS_PATH}/final_classical_results{DATA_SENTINEL}_exp_{experiment}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n Final results saved to {summary_path}")


# ======================================================
# Entry Point
# ======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate classical ML models with best hyperparameters.")
    parser.add_argument("--experiment", type=int, default=501, help="Experiment ID number.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    main(experiment=args.experiment, seed=args.seed)
