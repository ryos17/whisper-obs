import json
import matplotlib.pyplot as plt
import os

# Define file paths and corresponding model names and colors

model_files = {
    'model_compare_result/tiny_obs_new.json': {'name': 'whisper-tiny', 'color': 'blue', 'marker': 'o'},
    'model_compare_result/base_obs_new.json': {'name': 'whisper-base', 'color': 'green', 'marker': 's'},
    'model_compare_result/small_obs_new.json': {'name': 'whisper-small', 'color': 'red', 'marker': '^'},
    'model_compare_result/medium_obs_new.json': {'name': 'whisper-medium', 'color': 'purple', 'marker': 'd'},
    'model_compare_result/large_obs_new.json': {'name': 'whisper-large', 'color': 'orange', 'marker': 'x'}
}
title = 'Whisper Model Size Comparison with OBS Pruning'
file_name = 'model_compare_result/model_size_comparison.png'

# model_files = {
#     'model_compare_result/tiny_obs_new.json': {'name': 'obs', 'color': 'blue', 'marker': 'o'},
#     'model_compare_result/tiny_obs_finetune.json': {'name': 'obs-finetune', 'color': 'green', 'marker': 's'},
#     'model_compare_result/tiny_mp_global.json': {'name': 'mp-global', 'color': 'red', 'marker': '^'},
#     'model_compare_result/tiny_mp_local.json': {'name': 'mp-local', 'color': 'purple', 'marker': 'd'},
#     'model_compare_result/tiny_mp_finetune_global.json': {'name': 'mp-finetune-global', 'color': 'orange', 'marker': 'x'},
#     'model_compare_result/tiny_mp_finetune_local.json': {'name': 'mp-finetune-local', 'color': 'gray', 'marker': 'v'},
#     'model_compare_result/tiny_iobs.json': {'name': 'iobs', 'color': 'black', 'marker': 'h'},
#     'model_compare_result/tiny_imp_global.json': {'name': 'imp-global', 'color': 'pink', 'marker': 's'},
#     'model_compare_result/tiny_imp_local.json': {'name': 'imp-local', 'color': 'darkgreen', 'marker': 'd'},
# }
# title = 'Whisper Tiny Model Comparison with Different Pruning Methods'
# file_name = 'model_compare_result/pruning_method_comparison.png'

model_files = {
    'model_compare_result/test_obs.json': {'name': 'obs', 'color': 'blue', 'marker': 'o'},
    'model_compare_result/test_mp_global.json': {'name': 'mp-global', 'color': 'red', 'marker': '^'},
    'model_compare_result/test_mp_local.json': {'name': 'mp-local', 'color': 'purple', 'marker': 'd'},
    'model_compare_result/test_obs_finetune.json': {'name': 'obs-finetune', 'color': 'green', 'marker': 's'},
    'model_compare_result/test_mp_finetune_global.json': {'name': 'mp-finetune-global', 'color': 'orange', 'marker': 'x'},
    'model_compare_result/test_mp_finetune_local.json': {'name': 'mp-finetune-local', 'color': 'gray', 'marker': 'v'},
    'model_compare_result/test_iobs.json': {'name': 'iobs', 'color': 'black', 'marker': 'h'},
    'model_compare_result/test_imp_global.json': {'name': 'imp-global', 'color': 'pink', 'marker': 's'},
    'model_compare_result/test_imp_local.json': {'name': 'imp-local', 'color': 'darkgreen', 'marker': 'd'},
}
title = 'Whisper Tiny Model Comparison with Different Pruning Methods'
file_name = 'model_compare_result/test_pruning_method_comparison.png'

# model_files = {
#     'model_compare_result/obs_batch_1.json': {'name': 'obs-batch-1', 'color': 'blue', 'marker': 'o'},
#     'model_compare_result/obs_batch_4.json': {'name': 'obs-batch-4', 'color': 'green', 'marker': 's'},
#     'model_compare_result/obs_batch_16.json': {'name': 'obs-batch-16', 'color': 'purple', 'marker': 'd'},
#     'model_compare_result/obs_batch_32.json': {'name': 'obs-batch-32', 'color': 'darkgreen', 'marker': 'h'},
#     'model_compare_result/obs_batch_64.json': {'name': 'obs-batch-64', 'color': 'orange', 'marker': 'x'},
#     'model_compare_result/obs_batch_128.json': {'name': 'obs-batch-128', 'color': 'red', 'marker': '^'},
# }
# title = 'Whisper Tiny Model Comparison with Different Batch Sizes'
# file_name = 'model_compare_result/batch_size_comparison.png'

# Create plot
plt.figure(figsize=(12, 8))

# Process each model's data
for file_path, model_info in model_files.items():
    # Skip if file doesn't exist yet
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} does not exist yet. Skipping.")
        continue
    
    try:
        # Load results
        with open(file_path, 'r') as f:
            results = json.load(f)
        
        # Extract data
        sparsities = []
        normalized_wers = []
        
        for sparsity_str, metrics in results.items():
            sparsity = float(sparsity_str) * 100  # Convert to percentage
            norm_wer = min(metrics['normalized_wer'], 100)  # Cap at 100
            
            sparsities.append(sparsity)
            normalized_wers.append(norm_wer)
        
        # Sort by sparsity to ensure the line is drawn correctly
        sorted_data = sorted(zip(sparsities, normalized_wers))
        sparsities = [x for x, y in sorted_data]
        normalized_wers = [y for x, y in sorted_data]
        
        # Plot this model's data
        plt.plot(
            sparsities, 
            normalized_wers, 
            color=model_info['color'],
            marker=model_info['marker'],
            linestyle='-', 
            linewidth=2, 
            markersize=8,
            label=model_info['name']
        )
        
        print(f"Plotted data from {file_path}")
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Set labels and title with larger font sizes
plt.xlabel('Sparsity (%)', fontsize=16)
plt.ylabel('Normalized WER (%)', fontsize=16)
plt.title(title, fontsize=18)

# Set x-axis to show every 10% and set limits to hit edges
plt.xticks(range(0, 101, 10), fontsize=14)
plt.xlim(0, 100)

# Set y-axis limits with some padding
plt.ylim(0, 100)
plt.yticks(fontsize=14)

# Add grid
plt.grid(True, alpha=0.3)

# Add legend
plt.legend(fontsize=14, loc='best')

# Save plot
plt.tight_layout()
plt.savefig(file_name, dpi=300, bbox_inches='tight')

# Show plot
plt.show()

print(f"Plot saved as {file_name}")
