#!/bin/bash

# ===============================
# Bash Script to Run All Model Searches
# ===============================

# Experiment and Trial Configuration

#!/bin/bash

EXPERIMENT_ID=200
FT=false
TARGET_LAYERS_STRINGS=("middle" "" "last")
SELECTOR_IDX=(0 1 2)
MODELSTR="resnet50"

for idx in "${!TARGET_LAYERS_STRINGS[@]}"; do
    TARGET_LAYERS_STRING="${TARGET_LAYERS_STRINGS[$idx]}"
    SELECTOR="${SELECTOR_IDX[$idx]}"

    echo "Running saliency.py for TARGET_LAYERS_STRING='${TARGET_LAYERS_STRING}', SELECTOR_IDX=${SELECTOR}, model_name=CustomCNNModel, modelstr=dense121"
    python saliency.py --experiment ${EXPERIMENT_ID} --model_name CustomCNNModel --modelstr dense121 --ft ${FT} --target_layers_string "${TARGET_LAYERS_STRING}" --selector_idx ${SELECTOR}

    echo "Running saliency.py for TARGET_LAYERS_STRING='${TARGET_LAYERS_STRING}', SELECTOR_IDX=${SELECTOR}, model_name=CustomCNNModel, modelstr=resnet18"
    python saliency.py --experiment ${EXPERIMENT_ID} --model_name CustomCNNModel --modelstr ${MODELSTR} --ft ${FT} --target_layers_string "${TARGET_LAYERS_STRING}" --selector_idx ${SELECTOR}

    echo "Running saliency.py for TARGET_LAYERS_STRING='${TARGET_LAYERS_STRING}', SELECTOR_IDX=${SELECTOR}, model_name=ViTModel, modelstr=dense121"
    python saliency.py --experiment ${EXPERIMENT_ID} --model_name ViTModel --modelstr dense121 --ft ${FT} --target_layers_string "${TARGET_LAYERS_STRING}" --selector_idx ${SELECTOR}

    echo "Running saliency.py for TARGET_LAYERS_STRING='${TARGET_LAYERS_STRING}', SELECTOR_IDX=${SELECTOR}, model_name=ViTModel, modelstr=resnet18"
    python saliency.py --experiment ${EXPERIMENT_ID} --model_name ViTModel --modelstr ${MODELSTR} --ft ${FT} --target_layers_string "${TARGET_LAYERS_STRING}" --selector_idx ${SELECTOR}

    echo "Running sal.py for TARGET_LAYERS_STRING='${TARGET_LAYERS_STRING}', SELECTOR_IDX=${SELECTOR}, model_name=CustomCNNModel, modelstr=dense121"
    python sal.py --experiment ${EXPERIMENT_ID} --model_name CustomCNNModel --modelstr dense121 --ft ${FT} --target_layers_string "${TARGET_LAYERS_STRING}" --selector_idx ${SELECTOR}

    echo "Running sal.py for TARGET_LAYERS_STRING='${TARGET_LAYERS_STRING}', SELECTOR_IDX=${SELECTOR}, model_name=CustomCNNModel, modelstr=resnet18"
    python sal.py --experiment ${EXPERIMENT_ID} --model_name CustomCNNModel --modelstr ${MODELSTR} --ft ${FT} --target_layers_string "${TARGET_LAYERS_STRING}" --selector_idx ${SELECTOR}

    echo "Running sal.py for TARGET_LAYERS_STRING='${TARGET_LAYERS_STRING}', SELECTOR_IDX=${SELECTOR}, model_name=ViTModel, modelstr=dense121"
    python sal.py --experiment ${EXPERIMENT_ID} --model_name ViTModel --modelstr dense121 --ft ${FT} --target_layers_string "${TARGET_LAYERS_STRING}" --selector_idx ${SELECTOR}

    echo "Running sal.py for TARGET_LAYERS_STRING='${TARGET_LAYERS_STRING}', SELECTOR_IDX=${SELECTOR}, model_name=ViTModel, modelstr=resnet18"
    python sal.py --experiment ${EXPERIMENT_ID} --model_name ViTModel --modelstr ${MODELSTR} --ft ${FT} --target_layers_string "${TARGET_LAYERS_STRING}" --selector_idx ${SELECTOR}
done


# SELECTOR_IDX=(0 1 2) # Index to select target layer from candidates
# python saliency.py --experiment ${EXPERIMENT_ID} --model_name CustomCNNModel --modelstr dense121 --ft false --target_layers_string ${TARGET_LAYERS_STRING} --selector_idx ${SELECTOR_IDX}
# python saliency.py --experiment ${EXPERIMENT_ID} --model_name CustomCNNModel --modelstr resnet18 --ft false --target_layers_string ${TARGET_LAYERS_STRING} --selector_idx ${SELECTOR_IDX}
# python saliency.py --experiment ${EXPERIMENT_ID} --model_name ViTModel --modelstr dense121 --ft false --target_layers_string ${TARGET_LAYERS_STRING} --selector_idx ${SELECTOR_IDX}
# python saliency.py --experiment ${EXPERIMENT_ID} --model_name ViTModel --modelstr resnet18 --ft false --target_layers_string ${TARGET_LAYERS_STRING} --selector_idx ${SELECTOR_IDX}

# python sal.py --experiment ${EXPERIMENT_ID} --model_name CustomCNNModel --modelstr dense121 --ft false --target_layers_string ${TARGET_LAYERS_STRING} --selector_idx ${SELECTOR_IDX}
# python sal.py --experiment ${EXPERIMENT_ID} --model_name CustomCNNModel --modelstr resnet18 --ft false --target_layers_string ${TARGET_LAYERS_STRING} --selector_idx ${SELECTOR_IDX}
# python sal.py --experiment ${EXPERIMENT_ID} --model_name ViTModel --modelstr dense121 --ft false --target_layers_string ${TARGET_LAYERS_STRING} --selector_idx ${SELECTOR_IDX}
# python sal.py --experiment ${EXPERIMENT_ID} --model_name ViTModel --modelstr resnet18 --ft false --target_layers_string ${TARGET_LAYERS_STRING} --selector_idx ${SELECTOR_IDX}

# ===============================
# Completion Message
# ===============================
echo "======================================"
echo "All Searches Completed Successfully!"
echo "======================================"