# Whisper Pruning

This repository implements multiple pruning techniques for Whisper speech recognition models, including Optimal Brain Surgeon (OBS) and Iterative Magnitude Pruning (IMP).

## Features

- **Multiple Pruning Techniques**: 
  - OBS (Optimal Brain Surgeon) with diagonal approximation
  - IMP (Iterative Magnitude Pruning) with local and global variants
  - Enhanced OBS with improved saliency using first-order information
  - Mixed sparsity pruning based on layer sensitivity

- **Comprehensive Evaluation**: Tools to evaluate pruned models on speech recognition tasks
  - WER (Word Error Rate) and CER (Character Error Rate) metrics
  - Normalized metrics using WhisperNormalizer
  - Evaluation on LibriSpeech dataset

- **GPU Support**: Multi-GPU support with configurable device selection
  - Compatible with CUDA devices
  - Memory-efficient implementation

- **Visualization**: Tools to plot and compare results from different pruning methods

## Key Components

### Pruning Classes

#### WhisperOBSPruner

```python
from utils.obs import WhisperOBSPruner, utility_obs_prune

# Initialize pruner with device selection
pruner = WhisperOBSPruner(model, device=0, debug=True)

# Prune model with mixed sparsity
pruned_model = utility_obs_prune(
    model=model,
    processor=processor,
    audio_path=audio_path,
    sparsity=0.3,
    device=0,
    alpha=0.1  # Controls mixed sparsity range
)
```

#### WhisperIMPPruner

```python
from utils.imp import WhisperIMPPruner, utility_imp_prune

# Prune model using global or local IMP
pruned_model = utility_imp_prune(
    model=model,
    processor=processor,
    audio_path=audio_path,
    sparsity=0.3,
    device=0,
    prune_method="global"  # or "local"
)
```

## Advanced Pruning Features

### Improved Saliency with First-order Information

OBS implementation includes an enhanced saliency metric that combines first and second-order information:

```
improved_saliency = |w*g| + (w² / H_inv_diag)
```

Where:
- w is the weight value
- g is the gradient of the loss with respect to the weight
- H_inv_diag is the inverted diagonal Hessian

### Mixed Sparsity Pruning

Assigns different sparsity levels to layers based on their sensitivity:

1. Computes layer sensitivities using Hessian trace estimation
2. Ranks layers by sensitivity (higher sensitivity = lower sparsity)
3. Assigns sparsity levels using a controlled range (target ± alpha)
4. Ensures overall target sparsity is precisely achieved

## Evaluation

```python
# Evaluate a pruned model
metrics = utility_obs_evaluate(
    model=pruned_model,
    processor=processor,
    num_samples=100,
    device=0
)

print(f"WER: {metrics['wer']:.2f}%")
print(f"CER: {metrics['cer']:.2f}%")
print(f"Normalized WER: {metrics['normalized_wer']:.2f}%")
print(f"Normalized CER: {metrics['normalized_cer']:.2f}%")
```

## Usage Examples

### Pruning and Evaluation Script

```python
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from utils.obs import utility_obs_prune, utility_obs_evaluate

# Load model
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

# Define sparsity levels to test
sparsities = [0.1, 0.3, 0.5, 0.7, 0.9]
results = {}

# Audio path for pruning
audio_path = "/path/to/audio/sample.flac"

# Test different sparsity levels
for sparsity in sparsities:
    # Prune model
    pruned_model = utility_obs_prune(
        model=model,
        processor=processor,
        audio_path=audio_path,
        sparsity=sparsity,
        device=0
    )
    
    # Evaluate pruned model
    metrics = utility_obs_evaluate(
        model=pruned_model,
        processor=processor,
        num_samples=100,
        device=0
    )
    
    # Store results
    results[sparsity] = metrics
    
    # Clean up GPU memory
    del pruned_model
    torch.cuda.empty_cache()
```

### Comparing Pruning Methods

The repository includes tools to compare different pruning methods:

```bash
python eval.py --model openai/whisper-tiny --method obs --sparsities 0.1,0.3,0.5,0.7,0.9 --output obs_results.json
python eval.py --model openai/whisper-tiny --method imp_global --sparsities 0.1,0.3,0.5,0.7,0.9 --output imp_global_results.json
python eval.py --model openai/whisper-tiny --method imp_local --sparsities 0.1,0.3,0.5,0.7,0.9 --output imp_local_results.json

python plot.py --files obs_results.json imp_global_results.json imp_local_results.json --output comparison.png
```

## Technical Details

### OBS Algorithm with Diagonal Approximation

1. **Hessian Trace Estimation**: Uses Hutchinson's algorithm for efficient trace estimation
   ```python
   # Hutchinson's method: trace(H) ≈ E[z^T H z] where z is random
   z = torch.randn(rows, cols, device=device)
   Hz = torch.zeros_like(z)
   # ... matrix operations ...
   trace_estimate += torch.sum(z * Hz).item()
   ```

2. **Efficient Pruning**: Processes weights in batches and chunks for memory efficiency
   ```python
   # Process rows in parallel batches
   for i1 in range(0, rows, batch_size):
       i2 = min(i1 + batch_size, rows)
       # ... pruning operations ...
   ```

### IMP Implementation

Supports both local (layer-wise) and global (model-wide) magnitude pruning:

- **Local IMP**: Prunes a fixed percentage of weights from each layer independently
- **Global IMP**: Prunes weights across all layers based on absolute magnitude

## Dependencies

- PyTorch
- Transformers
- TorchAudio
- NumPy
- Matplotlib
- Evaluate

## Notes

- GPU device selection is critical for large models - use the `device` parameter consistently
- Mixed sparsity pruning provides better results than uniform sparsity
- The improved saliency metric with first-order information enhances pruning quality
- For best results with large models (base/medium/large), ensure sufficient GPU memory
