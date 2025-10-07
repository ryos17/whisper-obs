#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt

# Load results
with open('pruned_model_evaluation_results.json', 'r') as f:
    results = json.load(f)

# Extract data
sparsities = []
normalized_wers = []

for sparsity_str, metrics in results.items():
    sparsity = float(sparsity_str) * 100  # Convert to percentage
    norm_wer = min(metrics['normalized_wer'], 100)  # Cap at 100
    
    sparsities.append(sparsity)
    normalized_wers.append(norm_wer)

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(sparsities, normalized_wers, 'bo-', linewidth=2, markersize=6)

# Set labels and title with larger font sizes
plt.xlabel('Sparsity (%)', fontsize=14)
plt.ylabel('Normalized WER (%)', fontsize=14)
plt.title('Sparsity vs Normalized WER for openai/whisper-tiny', fontsize=16)

# Set x-axis to show every 10% and set limits to hit edges
plt.xticks(range(0, 101, 10), fontsize=12)
plt.xlim(0, 100)

# Set y-axis limits
plt.ylim(0, 100)

# Add grid
plt.grid(True, alpha=0.3)

# Save plot
plt.tight_layout()
plt.savefig('sparsity_vs_normalized_wer.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plot saved as 'sparsity_vs_normalized_wer.png'")
