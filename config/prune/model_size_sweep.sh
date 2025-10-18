#!/bin/bash

# Model size sweep script for iobs method
# Iterates through different Whisper model sizes: medium, base, small, large

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Create output directory if it doesn't exist
mkdir -p result/model_size_iobs_sweep

# Array of model sizes to test
sizes=("base" "small" "medium" "large")

# Loop through each model size
for size in "${sizes[@]}"; do
    echo "Running evaluation for whisper-${size} with iobs method..."

    # Run the evaluation
    python eval.py \
        --method iobs \
        --model "openai/whisper-${size}" \
        --output "result/model_size_iobs_sweep/${size}_iobs.json"
    
    echo "Completed evaluation for whisper-${size}"
    echo "Results saved to: result/model_size_iobs_sweep/${size}_iobs.json"
    echo "----------------------------------------"
done

echo "All model size evaluations completed!"
