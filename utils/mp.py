import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchaudio
import copy
from tqdm import tqdm
from typing import Dict, List, Optional
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class WhisperMPPruner:
    """
    MP Pruner for Whisper models.
    
    This class implements the Magnitude Pruning algorithm for pruning
    Linear layers in Whisper models with minimal performance degradation.
    """
    
    def __init__(self, model: WhisperForConditionalGeneration, device: int = 0, debug: bool = False):
        """
        Initialize the MP pruner for a Whisper model.
        
        Args:
            model: The Whisper model to be pruned
            device: Device to run on (default: 0)
            debug: Whether to print debug information (default: False)
        """
        self.model = model
        self.device = f"cuda:{device}"
        self.debug = debug
        
        # Store layer information
        self.layers_info: Dict[str, Dict] = {}
        self._collect_layers_info()
        
    def _collect_layers_info(self):
        """Collect information about all Linear layers in the model."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self.layers_info[name] = {
                    'layer': module,
                    'weight': module.weight.data.clone(),
                    'shape': module.weight.data.shape,
                    'n_params': module.weight.data.numel()
                }
                
                if self.debug:
                    print(f"\tRegistered layer {name} with shape {module.weight.data.shape}")
    
    def prune_layer_local(self, layer_name: str, sparsity: float) -> None:
        """
        Prune a single Linear layer using magnitude pruning (local pruning).
        
        Args:
            layer_name: Name of the layer to prune
            sparsity: Target sparsity (fraction of weights to remove)
        """
        if layer_name not in self.layers_info:
            raise ValueError(f"Layer {layer_name} not found in model")
            
        data = self.layers_info[layer_name]
        layer = data['layer']
        W = layer.weight.data.clone().to(self.device)
        
        # Check if there's already a weight mask from previous pruning
        if hasattr(layer, 'weight_mask'):
            mask = layer.weight_mask.clone().to(self.device)
            current_sparsity = 1.0 - (mask == 1).sum().item() / mask.numel()
            if self.debug:
                print(f"\tUsing existing weight mask, current sparsity: {current_sparsity:.4f}")
        else:
            mask = torch.ones_like(W, dtype=torch.float32, device=self.device)
            current_sparsity = 0.0
            if self.debug:
                print("\tCreating new weight mask")
        
        # Calculate how many more weights need to be pruned
        total_weights = mask.numel()
        current_pruned = (mask == 0).sum().item()
        target_pruned = int(sparsity * total_weights)
        additional_pruned = target_pruned - current_pruned
        
        if additional_pruned <= 0:
            if self.debug:
                print(f"\tNo additional pruning needed (target: {sparsity:.4f}, current: {current_sparsity:.4f})")
            return
        
        # Calculate threshold for additional pruning
        remaining_weights = W[mask == 1]
        if len(remaining_weights) == 0:
            if self.debug:
                print("\tNo remaining weights to prune")
            return
            
        k = min(additional_pruned, len(remaining_weights))
        if k > 0:
            threshold = torch.kthvalue(torch.abs(remaining_weights), k).values
            
            # Create new mask for additional pruning
            abs_weights = torch.abs(W)
            new_mask = torch.gt(abs_weights, threshold).float().to(self.device)
            
            # Combine with existing mask
            mask = mask * new_mask
            
        # Apply the computed mask to the layer
        prune.custom_from_mask(layer, 'weight', mask)
        
        if self.debug:
            # Print sparsity achieved
            total_mask_elements = layer.weight_mask.numel()
            num_nonzero = (layer.weight_mask == 1).sum().item()
            sparsity_achieved = 1.0 - (num_nonzero / total_mask_elements)
            print(f"\t{layer_name:<50} | Sparsity: {sparsity_achieved:.4f}")
    
    def prune_model_local(self, sparsity: float, target_layers: Optional[List[str]] = None) -> None:
        """
        Prune multiple layers in the model using local pruning.
        
        Args:
            sparsity: Target sparsity for all layers
            target_layers: List of layer names to prune. If None, prunes all Linear layers.
        """
        if target_layers is None:
            target_layers = list(self.layers_info.keys())
        
        for layer_name in tqdm(target_layers, desc="Pruning indivisual layers", leave=False):
            if layer_name in self.layers_info:
                self.prune_layer_local(layer_name, sparsity)
    
    def prune_model_global(self, sparsity: float, target_layers: Optional[List[str]] = None) -> None:
        """
        Prune model using global magnitude pruning across all layers.
        
        Args:
            sparsity: Target sparsity (fraction of weights to remove)
            target_layers: List of layer names to prune. If None, prunes all Linear layers.
        """
        if target_layers is None:
            target_layers = list(self.layers_info.keys())
            
        # Filter layers_info to only include target_layers
        filtered_layers = {name: data for name, data in self.layers_info.items() if name in target_layers}
        
        # Collect all weights into a single tensor for global threshold calculation
        all_weights = []
        all_masks = []
        for name, data in filtered_layers.items():
            layer = data['layer']
            W = layer.weight.data.clone().to(self.device)
            
            # Check if there's already a weight mask from previous pruning
            if hasattr(layer, 'weight_mask'):
                mask = layer.weight_mask.clone().to(self.device)
            else:
                mask = torch.ones_like(W, dtype=torch.float32, device=self.device)
            
            all_weights.append(torch.abs(W).view(-1))
            all_masks.append(mask.view(-1))
            
        if not all_weights:
            return
            
        all_weights_flat = torch.cat(all_weights)
        all_masks_flat = torch.cat(all_masks)
        
        # Calculate target total weights to prune
        total_weights = all_weights_flat.numel()
        target_pruned = int(total_weights * sparsity)
        current_pruned = (all_masks_flat == 0).sum().item()
        additional_pruned = target_pruned - current_pruned
        
        if additional_pruned <= 0:
            if self.debug:
                print(f"\tNo additional pruning needed (target: {sparsity:.4f}, current: {current_pruned/total_weights:.4f})")
            return
        
        # Only consider weights that are not already pruned
        remaining_weights = all_weights_flat[all_masks_flat == 1]
        total_remaining = len(remaining_weights)
        
        if total_remaining == 0:
            if self.debug:
                print("\tNo remaining weights to prune")
            return
        
        # Calculate how many additional weights to prune from remaining weights
        k = min(additional_pruned, total_remaining)
        
        if k > 0:
            # Calculate global threshold for additional pruning
            threshold = torch.kthvalue(remaining_weights, k).values
            
            if self.debug:
                print(f"\tGlobal pruning threshold: {threshold.item():.6f} for additional {k} weights")
            
            # Apply threshold to each layer
            for name in tqdm(target_layers, desc="Pruning global layers", leave=False):
                if name in filtered_layers:
                    data = filtered_layers[name]
                    layer = data['layer']
                    W = layer.weight.data.clone().to(self.device)
                    
                    # Check if there's already a weight mask from previous pruning
                    if hasattr(layer, 'weight_mask'):
                        mask = layer.weight_mask.clone().to(self.device)
                    else:
                        mask = torch.ones_like(W, dtype=torch.float32, device=self.device)
                    
                    # Create new mask for additional pruning
                    abs_weights = torch.abs(W)
                    new_mask = torch.gt(abs_weights, threshold).float().to(self.device)
                    
                    # Combine with existing mask
                    mask = mask * new_mask
                    
                    # Apply the computed mask to the layer
                    prune.custom_from_mask(layer, 'weight', mask)
                    
                    if self.debug:
                        # Print sparsity achieved
                        total_mask_elements = layer.weight_mask.numel()
                        num_nonzero = (layer.weight_mask == 1).sum().item()
                        sparsity_achieved = 1.0 - (num_nonzero / total_mask_elements)
                        print(f"\t{name:<50} | Sparsity: {sparsity_achieved:.4f}")
    
    
    def get_model_info(self) -> Dict[str, Dict]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        info = {}
        info['initial_num_prunable_params'] = 0
        info['final_num_prunable_params'] = 0
        for _, layer in self.model.named_modules():
            if isinstance(layer, nn.Linear) and hasattr(layer, 'weight'):
                info['initial_num_prunable_params'] += layer.weight_mask.numel()
                info['final_num_prunable_params'] += (layer.weight_mask != 0).sum().item()
        return info
    
    def cleanup(self):
        """Clear cached data."""
        self.layers_info.clear()
        torch.cuda.empty_cache()


def utility_mp_prune(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    sparsity: float,
    audio_path: str = "/datasets/speech/LibriSpeech/dev-clean/3081/166546/3081-166546-0000.flac",
    prune_method: str = "global",
    device: int = 0,
    debug: bool = False,
):
    """
    Prune a Whisper model using Magnitude Pruning.
    
    Args:
        model: Whisper model to prune
        processor: Whisper processor
        sparsity: Sparsity level to prune
        audio_path: Audio path to test the model
        prune_method: Pruning method, either "global" or "local" (default: "global")
        device: Device to run on (default: 0)
        debug: Whether to print debug information (default: False)
        
    Returns:
        Pruned model
    """    
    # Create MP pruner
    if debug:
        print("-" * 60)
        print("Setting up MP pruner...")
    pruned_model = copy.deepcopy(model)
    pruned_model = pruned_model.to(f"cuda:{device}")
    pruner = WhisperMPPruner(pruned_model, device=device, debug=debug)

    # Prune the model
    if debug:
        print("-" * 60)
        print(f"Pruning model with {prune_method} magnitude pruning (target sparsity: {sparsity:.2%})...")
    if prune_method == "global":
        pruner.prune_model_global(sparsity)
    else:  
        pruner.prune_model_local(sparsity)
    
    # Test both models
    with torch.no_grad():
        # Load and process sample audio
        audio, _ = torchaudio.load(audio_path)
        input_features = processor(audio.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features
        # Move to GPU
        input_features = input_features.to(f"cuda:{device}")
        model = model.to(f"cuda:{device}")
        # Generate and decode output
        original_output = model.generate(input_features, max_length=100, language="english", task="transcribe")
        original_text = processor.batch_decode(original_output, skip_special_tokens=True)[0]
        pruned_output = pruned_model.generate(input_features, max_length=100, language="english", task="transcribe")
        pruned_text = processor.batch_decode(pruned_output, skip_special_tokens=True)[0]
    print("-" * 60)
    print(f"{'Original output:':<30} {original_text[:30]}")
    print(f"{'Pruned output:':<30} {pruned_text[:30]}")
    
    # Calculate model size reduction
    layer_info = pruner.get_model_info()
    initial_prunable = layer_info['initial_num_prunable_params']
    final_prunable = layer_info['final_num_prunable_params']
    calculated_sparsity = 1 - (final_prunable / initial_prunable)
    print(f"{'Prunable weights (initial)':<30} {initial_prunable:,}")
    print(f"{'Prunable weights (final)':<30} {final_prunable:,}")
    print(f"{'Weights pruned:':<30} {initial_prunable - final_prunable:,}")
    print(f"{'Calculated sparsity:':<30} {calculated_sparsity:.2%}")
    
    # Cleanup
    if debug:
        print("-" * 60)
        print("Cleanup and finishing...")
    pruner.cleanup()

    # Move everything back to CPU
    model = model.to("cpu")
    pruned_model = pruned_model.to("cpu")
    input_features = input_features.to("cpu")
    torch.cuda.empty_cache()
    
    # Return the pruned model
    return pruned_model
