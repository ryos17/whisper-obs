# Whisper Pruning

This repository implements multiple pruning techniques for Whisper speech recognition models, including Optimal Brain Surgeon (OBS), Iterative Optimal Brain Surgeon (IOBS), Magnitude Pruning (MP), and Iterative Magnitude Pruning (IMP) with fine-tuning capabilities.


## Key Components

### Pruning Methods

#### 1. OBS (Optimal Brain Surgeon)
```python
from utils.obs import utility_obs_prune

# Basic OBS pruning
pruned_model = utility_obs_prune(
    model=model,
    processor=processor,
    audio_path=audio_path,
    sparsity=0.3,
    device=0,
    alpha=0.03  # Controls mixed sparsity range
)
```

#### 2. IOBS (Iterative OBS with Fine-tuning)
```python
from utils.iobs import utility_iobs_prune

# OBS with fine-tuning between pruning steps
pruned_model = utility_iobs_prune(
    model=model,
    processor=processor,
    audio_path=audio_path,
    sparsities=[0.1, 0.2, 0.3],  # Progressive sparsity levels
    device=0,
    epochs=2,  # Fine-tuning epochs
    learning_rate=3.75e-5
)
```

#### 3. MP (Magnitude Pruning)
```python
from utils.mp import utility_mp_prune

# Global magnitude pruning
pruned_model = utility_mp_prune(
    model=model,
    processor=processor,
    audio_path=audio_path,
    sparsity=0.3,
    prune_method="global",  # or "local"
    device=0
)
```

#### 4. IMP (Iterative Magnitude Pruning with Fine-tuning)
```python
from utils.imp import utility_imp_prune

# MP with fine-tuning between pruning steps
pruned_model = utility_imp_prune(
    model=model,
    processor=processor,
    audio_path=audio_path,
    sparsities=[0.1, 0.2, 0.3],  # Progressive sparsity levels
    prune_method="global",  # or "local"
    device=0,
    epochs=2,  # Fine-tuning epochs
    learning_rate=3.75e-5
)
```


## Evaluation

### Command Line Evaluation

The repository includes a comprehensive evaluation script that supports all pruning methods:

```bash
# Evaluate OBS pruning
python eval.py --model openai/whisper-tiny --method obs --sparsities 0.0,0.1,0.2,0.3,0.4,0.5 --output obs_results.json

# Evaluate OBS with fine-tuning
python eval.py --model openai/whisper-tiny --method obs_finetune --sparsities 0.0,0.1,0.2,0.3,0.4,0.5 --output obs_finetune_results.json

# Evaluate IOBS (Iterative OBS)
python eval.py --model openai/whisper-tiny --method iobs --sparsities 0.0,0.1,0.2,0.3,0.4,0.5 --output iobs_results.json

# Evaluate MP (Magnitude Pruning)
python eval.py --model openai/whisper-tiny --method mp_global --sparsities 0.0,0.1,0.2,0.3,0.4,0.5 --output mp_global_results.json
python eval.py --model openai/whisper-tiny --method mp_local --sparsities 0.0,0.1,0.2,0.3,0.4,0.5 --output mp_local_results.json

# Evaluate MP with fine-tuning
python eval.py --model openai/whisper-tiny --method mp_finetune_global --sparsities 0.0,0.1,0.2,0.3,0.4,0.5 --output mp_finetune_global_results.json
python eval.py --model openai/whisper-tiny --method mp_finetune_local --sparsities 0.0,0.1,0.2,0.3,0.4,0.5 --output mp_finetune_local_results.json

# Evaluate IMP (Iterative Magnitude Pruning)
python eval.py --model openai/whisper-tiny --method imp_global --sparsities 0.0,0.1,0.2,0.3,0.4,0.5 --output imp_global_results.json
python eval.py --model openai/whisper-tiny --method imp_local --sparsities 0.0,0.1,0.2,0.3,0.4,0.5 --output imp_local_results.json
```

### Evaluation Parameters

- `--model`: Whisper model to use (default: openai/whisper-tiny)
- `--method`: Pruning method (obs, obs_finetune, iobs, mp_local, mp_global, mp_finetune_local, mp_finetune_global, imp_local, imp_global)
- `--sparsities`: Comma-separated list of sparsity levels (default: 0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0)
- `--num-samples`: Number of evaluation samples (default: 100)
- `--device`: GPU device ID (default: 0)
- `--debug`: Enable debug output

## Usage Examples

### Complete Evaluation Workflow

```bash
# 1. Run evaluations for different methods
python eval.py --model openai/whisper-tiny --method obs --sparsities 0.0,0.1,0.2,0.3,0.4,0.5 --output model_compare_result/tiny_obs_new.json
python eval.py --model openai/whisper-tiny --method obs_finetune --sparsities 0.0,0.1,0.2,0.3,0.4,0.5 --output model_compare_result/tiny_obs_finetune.json
python eval.py --model openai/whisper-tiny --method mp_global --sparsities 0.0,0.1,0.2,0.3,0.4,0.5 --output model_compare_result/tiny_mp_global.json
python eval.py --model openai/whisper-tiny --method mp_local --sparsities 0.0,0.1,0.2,0.3,0.4,0.5 --output model_compare_result/tiny_mp_local.json

# 2. Generate comparison plots
python plot.py
```

### Visualization

The `plot.py` script automatically generates comparison plots from the evaluation results:

```python
# The plot.py script compares different methods:
model_files = {
    'model_compare_result/tiny_obs_new.json': {'name': 'obs', 'color': 'blue', 'marker': 'o'},
    'model_compare_result/tiny_obs_finetune.json': {'name': 'obs-finetune', 'color': 'green', 'marker': 's'},
    'model_compare_result/tiny_mp_global.json': {'name': 'mp-global', 'color': 'red', 'marker': '^'},
    'model_compare_result/tiny_mp_local.json': {'name': 'mp-local', 'color': 'purple', 'marker': 'd'},
}
```


## Technical Details

### Pruning Methods Overview

#### 1. OBS (Optimal Brain Surgeon)
- **Hessian Trace Estimation**: Uses Hutchinson's algorithm for efficient trace estimation
- **Improved Saliency**: Combines first and second-order information: `|w*g| + (w² / H_inv_diag)`
- **Mixed Sparsity**: Assigns different sparsity levels based on layer sensitivity
- **Memory Efficient**: Processes weights in batches and chunks

#### 2. IOBS (Iterative OBS)
- **Progressive Pruning**: Applies OBS pruning in multiple steps
- **Fine-tuning Integration**: Fine-tunes model between pruning steps
- **Better Performance**: Maintains model quality through iterative refinement

#### 3. MP (Magnitude Pruning)
- **Local MP**: Prunes a fixed percentage of weights from each layer independently
- **Global MP**: Prunes weights across all layers based on absolute magnitude
- **Simple and Fast**: Efficient implementation with minimal computational overhead

#### 4. IMP (Iterative Magnitude Pruning)
- **Progressive Pruning**: Applies MP pruning in multiple steps
- **Fine-tuning Integration**: Fine-tunes model between pruning steps
- **Combines Benefits**: Merges efficiency of MP with quality preservation of fine-tuning


## Dependencies

- PyTorch
- Transformers
- TorchAudio
- NumPy
- Matplotlib
- Evaluate
- Datasets
- Tqdm

## File Structure

```
whisper-obs/
├── eval.py                    # Main evaluation script
├── plot.py                    # Visualization script
├── utils/
│   ├── obs.py                 # OBS pruning implementation
│   ├── iobs.py                 # Iterative OBS with fine-tuning
│   ├── mp.py                   # Magnitude pruning
│   ├── imp.py                  # Iterative magnitude pruning
│   └── __init__.py
├── model_compare_result/      # Evaluation results directory
└── README.md
```

