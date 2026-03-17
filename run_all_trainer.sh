#!/bin/bash

# ===============================
# Bash Script to Run All Model Searches
# ===============================

# Experiment and Trial Configuration

#!/bin/bash

EXPERIMENT_ID=100
TARGET_CLASSES='macaque'

python train_with_best_hyperparams.py --experiment ${EXPERIMENT_ID} --model_name CustomCNNModel --modelstr dense121 --target_class ${TARGET_CLASSES}
python train_with_best_hyperparams.py --experiment ${EXPERIMENT_ID} --model_name CustomCNNModel --modelstr resnet18 --target_class ${TARGET_CLASSES}
python train_with_best_hyperparams.py --experiment ${EXPERIMENT_ID} --model_name ViTModel --modelstr dense121 --target_class ${TARGET_CLASSES}
python train_with_best_hyperparams.py --experiment ${EXPERIMENT_ID} --model_name ViTModel --modelstr resnet18 --target_class ${TARGET_CLASSES}
python train_with_best_hyperparams.py --experiment ${EXPERIMENT_ID} --model_name ViTModel --modelstr resnet18 --target_class ${TARGET_CLASSES} --pretrained True
python train_with_best_hyperparams.py --experiment ${EXPERIMENT_ID} --model_name SmallResCNNv5 --modelstr resnet18 --target_class ${TARGET_CLASSES}
python train_with_best_hyperparams.py --experiment ${EXPERIMENT_ID} --model_name SmallResCNNv5 --modelstr dense121 --target_class ${TARGET_CLASSES}

echo "======================================"
echo "All Searches Completed Successfully!"
echo "======================================"