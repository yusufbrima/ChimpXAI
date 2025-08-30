#!/bin/bash

# Fixed parameters
experiment=1
target_class="chimpanzee_ir"
duration=2

# Parameters to explore
methods=("triplet" "supcon")
ft_options=("True" "False")

# Loop through methods and ft options
for method in "${methods[@]}"; do
    for ft in "${ft_options[@]}"; do
        echo "Running with method=$method and ft=$ft"
        python saliency.py \
            --ft "$ft" \
            --experiment "$experiment" \
            --target_class "$target_class" \
            --contrastive_method "$method" \
            --duration "$duration"
    done
done