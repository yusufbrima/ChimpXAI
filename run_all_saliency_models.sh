#!/bin/bash

# ===============================
# Bash Script to Run All Model Searches
# ===============================

# Experiment and Trial Configuration

#!/bin/bash

EXPERIMENT_ID=100
FT=false
# TARGET_LAYERS_STRINGS=("middle" "" "last")
TARGET_LAYERS_STRINGS=("middle")
SELECTOR_IDX=(0)

# SELECTOR_IDX=(0 1 2)
MODELSTR="dense121"

for idx in "${!TARGET_LAYERS_STRINGS[@]}"; do
    TARGET_LAYERS_STRING="${TARGET_LAYERS_STRINGS[$idx]}"
    SELECTOR="${SELECTOR_IDX[$idx]}"

    # echo "Running saliency.py for TARGET_LAYERS_STRING='${TARGET_LAYERS_STRING}', SELECTOR_IDX=${SELECTOR}, model_name=CustomCNNModel, modelstr=dense121"
    python saliency.py --experiment ${EXPERIMENT_ID} --model_name CustomCNNModel --modelstr dense121 --ft ${FT} --target_layers_string "${TARGET_LAYERS_STRING}" --selector_idx ${SELECTOR}

    # echo "Running saliency.py for TARGET_LAYERS_STRING='${TARGET_LAYERS_STRING}', SELECTOR_IDX=${SELECTOR}, model_name=CustomCNNModel, modelstr=resnet18"
    python saliency.py --experiment ${EXPERIMENT_ID} --model_name CustomCNNModel --modelstr resnet18 --ft ${FT} --target_layers_string "${TARGET_LAYERS_STRING}" --selector_idx ${SELECTOR}

    echo "Running sal.py for TARGET_LAYERS_STRING='${TARGET_LAYERS_STRING}', SELECTOR_IDX=${SELECTOR}, model_name=CustomCNNModel, modelstr=dense121"
    python sal.py --experiment ${EXPERIMENT_ID} --model_name CustomCNNModel --modelstr dense121 --ft ${FT} --target_layers_string "${TARGET_LAYERS_STRING}" --selector_idx ${SELECTOR}

    echo "Running sal.py for TARGET_LAYERS_STRING='${TARGET_LAYERS_STRING}', SELECTOR_IDX=${SELECTOR}, model_name=CustomCNNModel, modelstr=resnet18"
    python sal.py --experiment ${EXPERIMENT_ID} --model_name CustomCNNModel --modelstr resnet18 --ft ${FT} --target_layers_string "${TARGET_LAYERS_STRING}" --selector_idx ${SELECTOR}

    python sal.py --experiment ${EXPERIMENT_ID} --model_name ViTModel --modelstr resnet18 --ft ${FT} --target_layers_string "${TARGET_LAYERS_STRING}" --selector_idx ${SELECTOR} --pretrained False
    python sal.py --experiment ${EXPERIMENT_ID} --model_name ViTModel --modelstr resnet18 --ft ${FT} --target_layers_string "${TARGET_LAYERS_STRING}" --selector_idx ${SELECTOR} --pretrained True

    python sal.py --experiment ${EXPERIMENT_ID} --model_name ViTModel --modelstr dense121 --ft ${FT} --target_layers_string "${TARGET_LAYERS_STRING}" --selector_idx ${SELECTOR} --pretrained False
    python sal.py --experiment ${EXPERIMENT_ID} --model_name ViTModel --modelstr dense121 --ft ${FT} --target_layers_string "${TARGET_LAYERS_STRING}" --selector_idx ${SELECTOR} --pretrained True

    python sal_sm.py --experiment ${EXPERIMENT_ID} --model_name SmallResCNNv5 --modelstr resnet18 
    python sal_sm.py --experiment ${EXPERIMENT_ID} --model_name SmallResCNNv5 --modelstr dense121 
    # python saliency.py --experiment ${EXPERIMENT_ID} --model_name ViTModel --modelstr resnet18 --ft ${FT} --target_layers_string "${TARGET_LAYERS_STRING}" --selector_idx ${SELECTOR} --pretrained True
    # python saliency.py --experiment ${EXPERIMENT_ID} --model_name ViTModel --modelstr resnet18 --ft ${FT} --target_layers_string "${TARGET_LAYERS_STRING}" --selector_idx ${SELECTOR} --pretrained False
done



# python train_with_best_hyperparams.py --experiment 200  
# ===============================
# Completion Message
# ===============================
echo "======================================"
echo "All Searches Completed Successfully!"
echo "======================================"