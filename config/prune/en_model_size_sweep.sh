#!/bin/bash

# Model size sweep script for iobs method
# Iterates through different Whisper model sizes: medium, base, small, large

# Set CUDA device
export CUDA_VISIBLE_DEVICES=1

# Create output directory if it doesn't exist
mkdir -p result/en_model_size_iobs_sweep

# Array of model sizes to test
sizes=("tiny" "base" "small" "medium")

# Loop through each model size
for size in "${sizes[@]}"; do
    echo "Running evaluation for whisper-${size}.en with iobs method..."

    # Run the evaluation
    python eval.py \
        --method iobs \
        --model "openai/whisper-${size}.en" \
        --output "result/en_model_size_iobs_sweep/${size}_en_iobs.json" \
        --en_model
    
    echo "Completed evaluation for whisper-${size}.en"
    echo "Results saved to: result/en_model_size_iobs_sweep/${size}_en_iobs.json"
    echo "----------------------------------------"
done

echo "All model size evaluations completed!"
