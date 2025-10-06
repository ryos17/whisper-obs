import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from utils.obs import WhisperOBSPruner

if __name__ == "__main__":   
    print("-" * 60) 
    print("Loading Whisper model...")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    
    # Load sample audio
    print("-" * 60)
    print("Loading sample audio...")
    audio, _ = torchaudio.load("/datasets/speech/LibriSpeech/dev-clean/3081/166546/3081-166546-0000.flac")
    
    # Process audio
    inputs = processor(audio.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features
    
    # Test original model
    print("-" * 60)
    print("Testing original model...")
    with torch.no_grad():
        original_output = model.generate(input_features, max_length=100)
        original_text = processor.batch_decode(original_output, skip_special_tokens=True)[0]
        print(f"Original output: {original_text}")
    
    # Create OBS pruner
    print("-" * 60)
    print("Setting up OBS pruner...")
    pruner = WhisperOBSPruner(model, debug=True)
    
    # Run forward pass to collect data
    print("-" * 60)
    print("Collecting data for Hessian computation...")
    with torch.no_grad():
        _ = model.generate(input_features, max_length=100)
    
    # Accumulate Hessian matrices
    print("-" * 60)
    print("Accumulating Hessian matrices...")
    pruner.accumulate_hessian(max_samples=100, batch_size=128)
    
    # Get layer information
    print("-" * 60)
    print("Pruning model...")
    sparsity = 0.95
    pruned_weights = pruner.prune_model(sparsity, batch_size=128)
    
    # Test pruned model
    print("-" * 60)
    with torch.no_grad():
        pruned_output = model.generate(input_features, max_length=100)
        pruned_text = processor.batch_decode(pruned_output, skip_special_tokens=True)[0]
        print(f"{'Original output:':<20} {original_text}")
        print(f"{'Pruned output:':<20} {pruned_text}")
    
    # Calculate model size reduction
    total_params = sum(p.numel() for p in model.parameters())
    non_zero_params = sum((p != 0).sum().item() for p in model.parameters())
    actual_sparsity = 1 - (non_zero_params / total_params)
    print(f"{'=' * 60}")
    print(f"{'MODEL STATISTICS':^60}")
    print(f"{'=' * 60}")
    print(f"{'Total parameters:':<25} {total_params:,}")
    print(f"{'Non-zero parameters:':<25} {non_zero_params:,}")
    print(f"{'Parameters removed:':<25} {total_params - non_zero_params:,}")
    print(f"{'Sparsity:':<25} {actual_sparsity:.2%}")
    print(f"{'=' * 60}")
    
    # Cleanup
    print("-" * 60)
    print("Cleanup...")
    pruner.cleanup()

    # Done!!!
    print("-" * 60)
    print("Done!")
    print("-" * 60)