import torch
import torch.nn as nn
import torchaudio
import copy
import evaluate
from tqdm import tqdm
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from typing import Dict, List, Optional
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class WhisperOBSPruner:
    """
    OBS Pruner for Whisper models.
    
    This class implements the Optimal Brain Surgeon algorithm for pruning
    Linear layers in Whisper models with minimal performance degradation.
    """
    
    def __init__(self, model: WhisperForConditionalGeneration, device: int = 0, debug: bool = False):
        """
        Initialize the OBS pruner for a Whisper model.
        
        Args:
            model: The Whisper model to be pruned
            device: Device to run on (default: 0)
            debug: Whether to print debug information (default: False)
        """
        self.model = model
        self.device = f"cuda:{device}"
        self.debug = debug
        
        # Store OBS data for each Linear layer
        self.obs_data: Dict[str, Dict] = {}
        
        # Register hooks to collect input/output data
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks to collect input/output data for Hessian computation."""
        def create_hook(name):
            def hook_fn(module, input, output):
                if isinstance(module, nn.Linear):
                    if name not in self.obs_data:
                        self.obs_data[name] = {
                            'layer': module,
                            'inputs': [],
                            'outputs': [],
                            'hessian_diag': None,
                            'nsamples': 0
                        }
                    
                    # Account for different input shapes
                    if isinstance(input, tuple):
                        inp = input[0]
                    else:
                        inp = input
                    if len(inp.shape) == 3: 
                        inp = inp.reshape(-1, inp.shape[-1])
                    
                    # Store input and output for Hessian computation
                    self.obs_data[name]['inputs'].append(inp.detach())
                    self.obs_data[name]['outputs'].append(output.detach())
            
            return hook_fn
        
        # Register hooks for all Linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(create_hook(name))
                self.hooks.append(hook)
                if self.debug:
                    print(f"Registered hook for {name}")
    
    def accumulate_hessian(self, max_samples: int = 1000, batch_size: int = 32):
        """
        Accumulate diagonal Hessian matrices for all Linear layers.
        
        Args:
            max_samples: Maximum number of input vector samples to use for Hessian computation (default: 1000)
            batch_size: Batch size for Hessian computation (default: 32)
        """
        if self.debug:
            print("Accumulating diagonal Hessian matrices...")
        
        for name, data in self.obs_data.items():
            if not data['inputs']:
                continue
                
            layer = data['layer']
            inputs = torch.cat(data['inputs'][:max_samples], dim=0)
            
            # Get weight matrix
            W = layer.weight.data
            rows, cols = W.shape
            
            # Initialize diagonal Hessian 
            H_diag = torch.zeros(cols, device=self.device, dtype=torch.double)
            nsamples = 0
            
            # Process inputs in batches
            batch_size = min(batch_size, len(inputs))
            for i in range(0, len(inputs), batch_size):
                batch_inputs = inputs[i:i+batch_size]
                batch_size_actual = batch_inputs.shape[0]
                
                # Update running average of diagonal Hessian
                H_diag *= nsamples / (nsamples + batch_size_actual)
                nsamples += batch_size_actual
                
                # Accumulate diagonal Hessian: H_diag += (2/n) * diag(X^T * X)
                X_squared = torch.sum(batch_inputs ** 2, dim=0)     
                H_diag += (2.0 / nsamples) * X_squared.double()
            
            # Store accumulated data
            data['hessian_diag'] = H_diag  
            data['nsamples'] = nsamples
            data['weight'] = W.clone()
            
            if self.debug:
                print(f"Layer {name}: Diagonal Hessian shape {H_diag.shape}, samples: {nsamples}")
    
    def _invert_hessian_diag(self, H_diag: torch.Tensor, rel_damp: float = 1e-4, min_value: float = 1e-8) -> torch.Tensor:
        """
        Invert diagonal Hessian matrix.
        
        Args:
            H_diag: Diagonal Hessian vector
            rel_damp: Relative damping factor for numerical stability when inverting the Hessian (default: 1e-4)
            min_value: Minimum value for the diagonal Hessian (default: 1e-8)
            
        Returns:
            Inverted diagonal Hessian vector
        """
        # Add small regularization to avoid division by zero
        reg = rel_damp * torch.mean(H_diag)
        H_diag_reg = H_diag + reg
        
        # Handle zero or negative values
        H_diag_reg = torch.clamp(H_diag_reg, min=min_value)
        
        # Invert diagonal Hessian
        H_inv_diag = 1.0 / H_diag_reg

        return H_inv_diag
    
    def prune_layer(self, layer_name: str, sparsity: float, rel_damp: float = 1e-4, min_value: float = 1e-8, batch_size: int = 32) -> torch.Tensor:
        """
        Prune a single Linear layer using diagonal OBS algorithm.
        
        Args:
            layer_name: Name of the layer to prune
            sparsity: Target sparsity (fraction of weights to remove)
            rel_damp: Relative damping factor for numerical stability when inverting the Hessian (default: 1e-4)
            min_value: Minimum value for the diagonal Hessian (default: 1e-8)
            batch_size: Batch size for pruning (default: 32)
            
        Returns:
            Pruned weight matrix
        """    
        data = self.obs_data[layer_name]
        H_diag = data['hessian_diag']
        W = data['weight']
        
        # Invert diagonal Hessian 
        H_inv_diag = self._invert_hessian_diag(H_diag, rel_damp=rel_damp, min_value=min_value)
        
        # Initialize pruning
        rows, cols = W.shape
        target_zeros = int(sparsity * cols)
        
        # Create mask for already pruned weights
        mask = torch.zeros_like(W, dtype=torch.bool, device=self.device)
        
        # Initialize loss tracking
        losses = torch.zeros((rows, cols + 1), device=self.device)
        
        # Process rows in parallel batches
        for i1 in range(0, rows, batch_size):
            i2 = min(i1 + batch_size, rows)
            count = i2 - i1
            
            # Get current weight slice
            w = W[i1:i2, :].clone()
            m = mask[i1:i2, :]
            
            # Track which weights are already pruned
            range_idx = torch.arange(count, device=self.device)
            
            # Prune weights iteratively
            for zeros in range(1, target_zeros + 1):
                # Compute saliency scores: w^2 / H_inv_diag
                saliency_scores = (w ** 2) / H_inv_diag.unsqueeze(0)  
                saliency_scores[m] = float('inf')  
                
                # Find smallest saliency score in each row
                j = torch.argmin(saliency_scores, dim=1)
                losses[i1:i2, zeros] = saliency_scores[range_idx, j]
                
                # Update weights 
                w[range_idx, j] = 0
                
                # Mark weight as pruned
                m[range_idx, j] = True
            
            W[i1:i2, :] = w
            mask[i1:i2, :] = m
        
        if self.debug:
            # Compute normalized loss: current_loss / ||WX||
            all_inputs = torch.cat(data['inputs'], dim=0)
            with torch.no_grad():
                outputs = W @ all_inputs.T  
                output_magnitude = torch.norm(outputs, dim=1, keepdim=True)  
            
            # Normalize losses by output magnitude
            normalized_losses = losses / output_magnitude
            print(f"{layer_name:<50} | Normalized loss: {torch.sum(normalized_losses).item():.3f}")

        return W
    
    def prune_model(self, sparsity: float, target_layers: Optional[List[str]] = None, batch_size: int = 32) -> Dict[str, torch.Tensor]:
        """
        Prune multiple layers in the model.
        
        Args:
            sparsity: Target sparsity for all layers
            target_layers: List of layer names to prune. If None, prunes all Linear layers.
            batch_size: Batch size for pruning (default: 32)
        Returns:
            Dictionary mapping layer names to pruned weight matrices
        """
        if target_layers is None:
            target_layers = list(self.obs_data.keys())
        
        pruned_weights = {}

        for layer_name in target_layers:
            if layer_name in self.obs_data:
                pruned_weight = self.prune_layer(layer_name, sparsity, batch_size=batch_size)
                pruned_weights[layer_name] = pruned_weight
                
                # Update the actual layer weights
                layer = self.obs_data[layer_name]['layer']
                layer.weight.data = pruned_weight
        
        return pruned_weights
    
    def get_layer_info(self) -> Dict[str, Dict]:
        """
        Get information about all Linear layers in the model.
        
        Returns:
            Dictionary with layer information
        """
        info = {}
        for name, data in self.obs_data.items():
            if data['hessian_diag'] is not None:
                info[name] = {
                    'shape': data['weight'].shape,
                    'sparsity': (data['weight'] == 0).float().mean().item(),
                    'nsamples': data['nsamples']
                }
        return info
    
    def cleanup(self):
        """Remove hooks and clear cached data."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.obs_data.clear()
        torch.cuda.empty_cache()


def utility_obs_prune(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio_path: str,
    sparsity: float,
    batch_size: int = 128,
    device: int = 0,
    debug: bool = False,
):
    """
    Prune a Whisper model using the Optimal Brain Surgeon algorithm.
    
    Args:
        model: Whisper model to prune
        processor: Whisper processor
        audio_path: Path to the calibration audio file
        sparsity: Sparsity level to prune
        batch_size: Batch size for pruning (default: 128)
        device: Device to run on (default: 0)
        debug: Whether to print debug information (default: False)
    """
    # Load and processsample audio
    if debug:
        print("-" * 60)
        print("Loading sample audio...")
    audio, _ = torchaudio.load(audio_path)
    inputs = processor(audio.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features
    
    # Create OBS pruner
    if debug:
        print("-" * 60)
        print("Setting up OBS pruner...")
    pruned_model = copy.deepcopy(model)
    pruner = WhisperOBSPruner(pruned_model, device=device, debug=debug)
    
    # Run forward pass to collect data
    if debug:
        print("-" * 60)
        print("Collecting data for Hessian computation...")
    with torch.no_grad():
        # Move input_features to the same device as the model
        input_features = input_features.to(f"cuda:{device}")
        _ = pruned_model.generate(input_features, max_length=100, language="english", task="transcribe")
    
    # Accumulate Hessian matrices
    if debug:
        print("-" * 60)
        print("Accumulating Hessian matrices...")
    pruner.accumulate_hessian(max_samples=100, batch_size=batch_size)
    
    # Prune the model
    if debug:
        print("-" * 60)
        print("Pruning model...")
    pruner.prune_model(sparsity, batch_size=batch_size)

    # Test both models
    if debug:
        print("-" * 60)
        print("Running test inference on input audio...")
        with torch.no_grad():
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


def utility_obs_evaluate(model, processor, num_samples=None, device=0, debug=False):
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
