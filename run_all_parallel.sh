#!/bin/bash

# ===============================
# Bash Script to Run All Model Searches (Parallelized)
# ===============================

# Experiment and Trial Configuration
EXPERIMENT_ID=200
N_TRIALS_DEEP=50
N_TRIALS_CLASSICAL=20

# ===============================
# Model and Target Definitions
# ===============================
MODEL_NAMES=('CustomCNNModel' 'ViTModel')
MODEL_STRS=('resnet18' 'dense121')
TARGET_CLASSES=('chimp')

# ===============================
# 1. Deep Learning Models (CNN / ViT)
# ===============================
echo "======================================"
echo "Running Hyperparameter Search for Deep Learning Models (Parallel)"
echo "======================================"

for i in "${!MODEL_NAMES[@]}"; do
  MODEL_NAME=${MODEL_NAMES[$i]}
  MODEL_STR=${MODEL_STRS[$i]}

  for TARGET in "${TARGET_CLASSES[@]}"; do
    echo "Launching $MODEL_NAME ($MODEL_STR) for target class: $TARGET"

    (
      python optuna_search.py \
        --model_name "$MODEL_NAME" \
        --modelstr "$MODEL_STR" \
        --target_class "$TARGET" \
        --n_trials "$N_TRIALS_DEEP" \
        --experiment "$EXPERIMENT_ID"

      python train_with_best_hyperparams.py \
        --model_name "$MODEL_NAME" \
        --modelstr "$MODEL_STR" \
        --experiment "$EXPERIMENT_ID" \
        --target_class "$TARGET"
    ) &
  done
done

# Wait for all deep learning jobs to complete
wait
echo "Deep learning model searches completed."
echo

# ===============================
# 2. Classical Machine Learning Models
# ===============================
echo "======================================"
echo "Running Hyperparameter Search for Classical Models"
echo "======================================"

(
  python classical_model_optuna_all.py \
    --n_trials "$N_TRIALS_CLASSICAL" \
    --experiment "$EXPERIMENT_ID"

  python train_final_classical_models.py \
    --experiment "$EXPERIMENT_ID"
) &
wait
echo "Classical model searches completed."
echo

# ===============================
# 4. Completion Message
# ===============================
echo "======================================"
echo "All Searches Completed Successfully!"
echo "======================================"