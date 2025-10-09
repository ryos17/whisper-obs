import torch
import torch.nn as nn
import torchaudio
import copy
import evaluate
from tqdm import tqdm
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from typing import Dict, List, Optional, Tuple, Union
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class WhisperIMPPruner:
    """
    IMP Pruner for Whisper models.
    
    This class implements the Iterative Magnitude Pruning algorithm for pruning
    Linear layers in Whisper models with minimal performance degradation.
    """
    
    def __init__(self, model: WhisperForConditionalGeneration, device: int = 0, debug: bool = False):
        """
        Initialize the IMP pruner for a Whisper model.
        
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
                    print(f"Registered layer {name} with shape {module.weight.data.shape}")
    
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
        W = data['weight'].clone()
        
        # Calculate threshold for pruning
        abs_weights = torch.abs(W)
        k = int(W.numel() * sparsity)
        if k > 0:  # Only prune if k > 0
            threshold = torch.kthvalue(abs_weights.view(-1), k).values
            
            # Create binary mask (1 for weights to keep, 0 for weights to prune)
            mask = torch.gt(abs_weights, threshold).float().to(self.device)
            
            # Apply mask to weights
            W = W * mask
            
        # Update the actual layer weights
        layer = self.layers_info[layer_name]['layer']
        layer.weight.data = W
            
        if self.debug:
            n_pruned = (W == 0).sum().item()
            actual_sparsity = n_pruned / W.numel()
            print(f"{layer_name:<50} | Target: {sparsity:.2%}, Actual: {actual_sparsity:.2%}")
    
    def prune_model_local(self, sparsity: float, target_layers: Optional[List[str]] = None) -> None:
        """
        Prune multiple layers in the model using local pruning.
        
        Args:
            sparsity: Target sparsity for all layers
            target_layers: List of layer names to prune. If None, prunes all Linear layers.
        """
        if target_layers is None:
            target_layers = list(self.layers_info.keys())
        
        for layer_name in target_layers:
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
        for name, data in filtered_layers.items():
            all_weights.append(torch.abs(data['weight']).view(-1))
            
        if not all_weights:
            return {}
            
        all_weights_flat = torch.cat(all_weights)
        total_params = all_weights_flat.numel()
        k = int(total_params * sparsity)
        
        if k > 0:  # Only prune if k > 0
            # Calculate global threshold
            threshold = torch.kthvalue(all_weights_flat, k).values
            
            if self.debug:
                print(f"Global pruning threshold: {threshold.item():.6f} for sparsity {sparsity:.2%}")
            
            # Apply threshold to each layer
            for name in target_layers:
                if name in filtered_layers:
                    data = filtered_layers[name]
                    W = data['weight'].clone()
                    
                    # Create binary mask (1 for weights to keep, 0 for weights to prune)
                    mask = torch.gt(torch.abs(W), threshold).float().to(self.device)
                    
                    # Apply mask to weights
                    W = W * mask
                    
                    # Update the actual layer weights
                    layer = data['layer']
                    layer.weight.data = W
                    
                    if self.debug:
                        n_pruned = (W == 0).sum().item()
                        actual_sparsity = n_pruned / W.numel()
                        print(f"{name:<50} | Actual: {actual_sparsity:.2%}")
    
    
    def get_layer_info(self) -> Dict[str, Dict]:
        """
        Get information about all Linear layers in the model.
        
        Returns:
            Dictionary with layer information
        """
        info = {}
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Linear):
                weight = layer.weight.data
                info[name] = {
                    'shape': weight.shape,
                    'sparsity': (weight == 0).float().mean().item(),
                    'n_params': weight.numel()
                }
        return info
    
    def cleanup(self):
        """Clear cached data."""
        self.layers_info.clear()
        torch.cuda.empty_cache()


def utility_imp_prune(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio_path: str,
    sparsity: float,
    prune_method: str = "global",
    device: int = 0,
    debug: bool = False,
):
    """
    Prune a Whisper model using Magnitude Pruning.
    
    Args:
        model: Whisper model to prune
        processor: Whisper processor
        audio_path: Path to the calibration audio file
        sparsity: Sparsity level to prune
        prune_method: Pruning method, either "global" or "local" (default: "global")
        device: Device to run on (default: 0)
        debug: Whether to print debug information (default: False)
        
    Returns:
        Pruned model
    """
    # Move model to GPU
    if debug:
        print("-" * 60)
        print("Moving model to GPU...")
    model = model.to(f"cuda:{device}")
    
    # Load and process sample audio
    if debug:
        print("-" * 60)
        print("Loading sample audio...")
    audio, _ = torchaudio.load(audio_path)
    inputs = processor(audio.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features
    
    # Create IMP pruner
    if debug:
        print("-" * 60)
        print("Setting up IMP pruner...")
    pruned_model = copy.deepcopy(model)
    pruner = WhisperIMPPruner(pruned_model, device=device, debug=debug)
    
    # Prune the model
    if debug:
        print("-" * 60)
        print(f"Pruning model with {prune_method} magnitude pruning...")
        print(f"Target sparsity: {sparsity:.2%}")
    
    # Call the appropriate pruning method
    if prune_method == "global":
        pruner.prune_model_global(sparsity)
    else:  # local
        pruner.prune_model_local(sparsity)
    
    # Calculate actual sparsity
    total_zeros = sum((layer.weight.data == 0).sum().item() for name, layer in pruned_model.named_modules() 
                     if isinstance(layer, nn.Linear))
    total_params = sum(layer.weight.data.numel() for name, layer in pruned_model.named_modules() 
                      if isinstance(layer, nn.Linear))
    
    actual_sparsity = total_zeros / total_params
    
    if debug:
        print(f"Actual sparsity after pruning: {actual_sparsity:.2%}")
    
    # Test both models
    if debug:
        print("-" * 60)
        print("Running test inference on input audio...")
        with torch.no_grad():
            # Move input_features to the same device as the model
            input_features = input_features.to(f"cuda:{device}")
            original_output = model.generate(input_features, max_length=100, language="english", task="transcribe")
            original_text = processor.batch_decode(original_output, skip_special_tokens=True)[0]
            pruned_output = pruned_model.generate(input_features, max_length=100, language="english", task="transcribe")
            pruned_text = processor.batch_decode(pruned_output, skip_special_tokens=True)[0]
        print(f"{'Original output:':<20} {original_text}")
        print(f"{'Pruned output:':<20} {pruned_text}")
    
    # Calculate model size reduction
    initial_params = sum((p != 0).sum().item() for p in model.parameters())
    final_params = sum((p != 0).sum().item() for p in pruned_model.parameters())
    actual_sparsity = 1 - (final_params / initial_params)
    print("-" * 60)
    print(f"{'Initial parameters:':<25} {initial_params:,}")
    print(f"{'Final parameters:':<25} {final_params:,}")
    print(f"{'Parameters removed:':<25} {initial_params - final_params:,}")
    print(f"{'Actual sparsity:':<25} {actual_sparsity:.2%}")
    
    # Cleanup
    if debug:
        print("-" * 60)
        print("Cleanup and finishing...")
    pruner.cleanup()
    
    # Return the pruned model
    return pruned_model


def utility_imp_evaluate(model, processor, num_samples=None, device=0, debug=False):
    """
    Evaluate a model on a dataset and return WER and CER.
    
    Args:
        model: Whisper model to evaluate
        processor: Whisper processor
        num_samples: Number of samples to evaluate. None means all samples (default: None)
        device: Device to run on (default: 0)
        debug: Whether to print debug information (default: False)
        
    Returns:
        Dictionary with WER and CER metrics
    """
    # Move model to GPU
    if debug:
        print("-" * 60)
        print("Moving model to GPU...")
    model = model.to(f"cuda:{device}")

    # Load metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    whisper_norm = BasicTextNormalizer()

    # Load evaluation dataset from custom files
    if debug:
        print("-" * 60)
        print("Loading evaluation dataset...")
    with open("custom_data/LibriSpeech/test-clean/audio_paths", "r") as f:
        audio_paths = [line.strip().split(" ", 1)[1] for line in f.readlines()]
    with open("custom_data/LibriSpeech/test-clean/text", "r") as f:
        texts = [line.strip().split(" ", 1)[1] for line in f.readlines()]
    
    # Take first num_samples samples if specified
    if num_samples is not None:
        audio_paths = audio_paths[:num_samples]
        texts = texts[:num_samples]
    
    # Set decoder prompt for English transcription
    model.config.forced_decoder_ids = (
        processor.tokenizer.get_decoder_prompt_ids(task="transcribe", language="english")
    )
    
    predictions = []
    references = []
    norm_predictions = []
    norm_references = []
    
    # Process dataset item by item
    for i in tqdm(range(len(audio_paths)), desc=f"Evaluating on {num_samples} samples", leave=False):
        audio_path = audio_paths[i]
        ref_text = texts[i]
        
        # Load and process audio
        audio, _ = torchaudio.load(audio_path)
        audio_array = audio.squeeze().numpy()
        
        # Process audio and generate prediction
        inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        
        # Generate prediction
        with torch.no_grad():
            output = model.generate(
                input_features, 
                max_length=100,
                language="english",
                task="transcribe"
            )
            pred_text = processor.batch_decode(output, skip_special_tokens=True)[0]
    
        # Store results
        predictions.append(pred_text)
        references.append(ref_text)
        norm_predictions.append(whisper_norm(pred_text))
        norm_references.append(whisper_norm(ref_text))
    
    # Calculate metrics
    wer = wer_metric.compute(references=references, predictions=predictions)
    cer = cer_metric.compute(references=references, predictions=predictions)
    norm_wer = wer_metric.compute(references=norm_references, predictions=norm_predictions)
    norm_cer = cer_metric.compute(references=norm_references, predictions=norm_predictions)
    
    return {
        "wer": round(100 * wer, 10),
        "cer": round(100 * cer, 10),
        "normalized_wer": round(100 * norm_wer, 10),
        "normalized_cer": round(100 * norm_cer, 10)
    }
