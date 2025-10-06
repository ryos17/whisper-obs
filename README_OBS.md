# Whisper OBS Pruning

This module provides a simplified implementation of OBS (Optimal Brain Surgeon) pruning specifically designed for Whisper models. The implementation focuses on unstructured pruning of Linear layers using second-order information.

## Features

- **Pure OBS Algorithm**: Implements the original OBS algorithm without quantization or structured pruning
- **Diagonal Approximation**: Uses diagonal Hessian approximation for massive speedup (100x+ faster)
- **Whisper-Specific**: Optimized for Whisper model architecture with 65 Linear layers
- **Automatic Layer Detection**: Automatically identifies and prunes Linear layers
- **Hessian Accumulation**: Collects second-order information during forward passes
- **Flexible Pruning**: Supports pruning specific layers or all Linear layers
- **Memory Efficient**: Processes layers in parallel batches

## Key Components

### WhisperOBSPruner Class

The main class for OBS pruning of Whisper models:

```python
from whisper_obs_pruning import WhisperOBSPruner

# Initialize pruner
pruner = WhisperOBSPruner(model, rel_damp=1e-4)

# Collect data and accumulate Hessian matrices
pruner.accumulate_hessian(max_samples=1000)

# Prune specific layers
pruned_weights = pruner.prune_model(sparsity=0.3, target_layers=['model.encoder.layers.0.fc1'])
```

### Convenience Function

For simple use cases:

```python
from whisper_obs_pruning import prune_whisper_model

# Prune entire model
pruned_model = prune_whisper_model(
    model=model,
    sparsity=0.3,  # Remove 30% of weights
    max_samples=1000
)
```

## Algorithm Details

### OBS (Optimal Brain Surgeon) Algorithm with Diagonal Approximation

1. **Diagonal Hessian Accumulation**: Collects only diagonal elements (much faster!)
   ```python
   H_diag += (2/n) * sum(X^2, dim=0)  # Only diagonal elements
   ```

2. **Weight Importance Scoring**: Uses diagonal Hessian to compute importance
   ```python
   scores = w^2 / H_inv_diag  # Much simpler with diagonal approximation
   ```

3. **Simplified Pruning**: Removes least important weights (no matrix operations needed)
   ```python
   w[j] = 0  # Simply set to zero with diagonal approximation
   ```

### Key Features

- **Diagonal Approximation**: 100x+ speedup by using only diagonal Hessian elements
- **Memory Efficient**: O(n) memory instead of O(n²) for full Hessian
- **Numerical Stability**: Simple regularization for diagonal elements
- **Parallel Processing**: Processes multiple weight rows simultaneously
- **Dead Neuron Handling**: Properly handles zero-variance inputs

## Usage Examples

### Basic Pruning

```python
import torch
from transformers import WhisperForConditionalGeneration
from whisper_obs_pruning import WhisperOBSPruner

# Load model
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

# Create pruner
pruner = WhisperOBSPruner(model)

# Run forward pass to collect data
dummy_input = torch.randn(1, 80, 3000)
with torch.no_grad():
    _ = model.generate(dummy_input, max_length=100)

# Accumulate Hessian matrices
pruner.accumulate_hessian(max_samples=500)

# Prune to 30% sparsity
pruned_weights = pruner.prune_model(sparsity=0.3)

# Cleanup
pruner.cleanup()
```

### Targeted Layer Pruning

```python
# Prune only attention layers
attention_layers = [
    name for name in pruner.obs_data.keys() 
    if 'attn' in name and 'proj' in name
]

pruned_weights = pruner.prune_model(
    sparsity=0.4, 
    target_layers=attention_layers
)
```

### Layer Information

```python
# Get information about all layers
layer_info = pruner.get_layer_info()
for name, info in layer_info.items():
    print(f"{name}: {info['shape']}, sparsity: {info['sparsity']:.3f}")
```

## Whisper Model Structure

The implementation targets these Linear layers in Whisper:

- **Encoder Layers**: `fc1`, `fc2` (feed-forward networks)
- **Attention Layers**: `k_proj`, `v_proj`, `q_proj`, `out_proj`
- **Output Projection**: `proj_out` (final output layer)

Total: 65 Linear layers in Whisper-tiny model

## Performance Considerations

- **Memory Usage**: Hessian matrices are stored in double precision
- **Computation Time**: O(n³) complexity for matrix operations
- **Batch Processing**: Processes layers in parallel for efficiency
- **GPU Memory**: Uses CUDA memory management for large models

## Testing

Run the test script to verify functionality:

```bash
python test_whisper_obs.py
```

This will:
1. Load a Whisper model
2. Collect Hessian data
3. Prune the model
4. Test both original and pruned models
5. Display performance metrics

## Dependencies

- PyTorch
- Transformers
- TorchAudio (for audio processing)

## Notes

- The implementation focuses on unstructured pruning (individual weight removal)
- For structured pruning (N:M patterns), use the original `trueobs.py` implementation
- Hessian accumulation requires multiple forward passes with representative data
- Pruning is performed in-place on the model weights
