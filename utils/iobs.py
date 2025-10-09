import os
import copy
import torch
import subprocess
from typing import List, Optional, Dict, Union
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from utils.obs import utility_obs_prune, WhisperOBSPruner

class WhisperIOBSPruner:
    """
    Iterative OBS Pruner for Whisper models.
    This class applies OBS pruning followed by fine-tuning in multiple stages.
    """
    
    def __init__(self, model: WhisperForConditionalGeneration, device: int = 0, debug: bool = False):
        """
        Initialize the IOBS pruner.
        
        Args:
            model: The Whisper model to be pruned
            device: GPU device ID to use
            debug: Whether to print debug information
        """
        self.model = model
        self.device = device
        self.debug = debug
        self.mask = None  # Will store the pruning mask
        
    def update_mask(self):
        """Update the pruning mask based on current model weights."""
        if self.mask is None:
            # Initialize mask dictionary for all linear layers
            self.mask = {}
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    self.mask[name] = torch.ones_like(module.weight.data, dtype=torch.bool)
        
        # Update mask based on current zeros in the model
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and name in self.mask:
                self.mask[name] = self.mask[name] & (module.weight.data != 0)
    
    def apply_mask(self):
        """Apply the stored mask to ensure pruned weights stay at zero."""
        if self.mask is None:
            return
            
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and name in self.mask:
                module.weight.data = module.weight.data * self.mask[name]

def utility_iobs_prune(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio_path: str,
    sparsities: List[float],
    train_datasets: List[str],
    eval_datasets: List[str],
    output_dir: str = "iobs_model",
    device: int = 0,
    debug: bool = False,
    learning_rate: float = 1e-5,
    num_steps: int = 1000,
    train_batchsize: int = 16,
    eval_batchsize: int = 8,
    eval_save_steps: int = 200,
    alpha: float = 0.1,
) -> WhisperForConditionalGeneration:
    """
    Apply Iterative OBS pruning with fine-tuning between pruning stages.
    
    Args:
        model: Whisper model to prune
        processor: Whisper processor
        audio_path: Path to audio file for OBS pruning
        sparsities: List of sparsity levels to apply sequentially
        train_datasets: List of training dataset paths
        eval_datasets: List of evaluation dataset paths
        output_dir: Directory to save fine-tuned models
        device: GPU device ID to use
        debug: Whether to print debug information
        learning_rate: Learning rate for fine-tuning
        num_steps: Number of fine-tuning steps per stage
        train_batchsize: Batch size for training
        eval_batchsize: Batch size for evaluation
        eval_save_steps: Steps between evaluations
        alpha: Alpha parameter for OBS mixed sparsity
        
    Returns:
        The pruned and fine-tuned model
    """
    if debug:
        print("-" * 60)
        print(f"Starting Iterative OBS pruning with {len(sparsities)} stages")
        
    # Create a deep copy of the model to avoid modifying the original
    pruned_model = copy.deepcopy(model)
    
    # Initialize IOBS pruner
    iobs_pruner = WhisperIOBSPruner(pruned_model, device=device, debug=debug)
    
    # Process each sparsity level sequentially
    for i, sparsity in enumerate(sparsities):
        if debug:
            print("=" * 60)
            print(f"Stage {i+1}/{len(sparsities)}: Sparsity {sparsity:.1%}")
            
        # 1. Apply OBS pruning at current sparsity level
        if debug:
            print("-" * 60)
            print(f"Applying OBS pruning at sparsity {sparsity:.1%}")
            
        pruned_model = utility_obs_prune(
            model=pruned_model,
            processor=processor,
            audio_path=audio_path,
            sparsity=sparsity,
            device=device,
            debug=debug,
            alpha=alpha
        )
        
        # Update the pruning mask after OBS
        iobs_pruner.model = pruned_model
        iobs_pruner.update_mask()
        
        # 2. Fine-tune the pruned model
        if debug:
            print("-" * 60)
            print(f"Fine-tuning model at sparsity {sparsity:.1%}")
            
        # Create a temporary directory for this stage's output
        stage_output_dir = os.path.join(output_dir, f"stage_{i+1}_sparsity_{int(sparsity*100)}")
        os.makedirs(stage_output_dir, exist_ok=True)
        
        # Save the model for fine-tuning
        temp_model_dir = os.path.join(stage_output_dir, "pre_finetune")
        os.makedirs(temp_model_dir, exist_ok=True)
        pruned_model.save_pretrained(temp_model_dir)
        processor.save_pretrained(temp_model_dir)
        
        # Prepare fine-tuning command
        train_datasets_str = " ".join(train_datasets)
        eval_datasets_str = " ".join(eval_datasets)
        
        finetune_cmd = [
            "python", "train/fine-tune_on_custom_dataset.py",
            f"--model_name={temp_model_dir}",
            f"--train_datasets", *train_datasets,  # Properly handle list arguments
            f"--eval_datasets", *eval_datasets,
            f"--output_dir={stage_output_dir}",
            f"--learning_rate={learning_rate}",
            f"--train_strategy=steps",
            f"--num_steps={num_steps}",
            f"--train_batchsize={train_batchsize}",
            f"--eval_batchsize={eval_batchsize}",
            f"--eval_save_steps={eval_save_steps}",
        ]
        
        # Execute fine-tuning
        if debug:
            print(f"Running fine-tuning command: {' '.join(finetune_cmd)}")
        
        try:
            subprocess.run(finetune_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Fine-tuning failed with error: {e}")
            if debug:
                print("Continuing with pruning without fine-tuning for this stage")
        
        # Load the fine-tuned model
        pruned_model = WhisperForConditionalGeneration.from_pretrained(stage_output_dir)
        pruned_model = pruned_model.to(f"cuda:{device}")
        
        # Apply mask to ensure pruned weights remain zero
        iobs_pruner.model = pruned_model
        iobs_pruner.apply_mask()
        
        if debug:
            # Count non-zero parameters
            non_zero = sum(p.numel() - (p == 0).sum().item() for p in pruned_model.parameters())
            total = sum(p.numel() for p in pruned_model.parameters())
            actual_sparsity = 1 - (non_zero / total)
            print(f"After fine-tuning - Actual sparsity: {actual_sparsity:.2%} (target: {sparsity:.2%})")
    
    if debug:
        print("=" * 60)
        print("Iterative OBS pruning completed")
    
    return pruned_model


