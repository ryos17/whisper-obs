"""
Evaluate pruned Whisper models on a dataset and save results as JSON.

This script prunes a Whisper model at different sparsity levels,
evaluates each on a dataset, and saves the results (WER, CER) as JSON.
"""

import json
import torch
import torchaudio
import evaluate
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from utils.obs import utility_obs_prune
from tqdm import tqdm

# Load metrics
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")
whisper_norm = BasicTextNormalizer()

def normalize_text(text):
    """Normalize text using Whisper's normalizer."""
    return whisper_norm(text)

def evaluate_model(model, processor, audio_paths, texts, device=0):
    """
    Evaluate a model on a dataset and return WER and CER.
    
    Args:
        model: Whisper model to evaluate
        processor: Whisper processor
        audio_paths: List of audio file paths
        texts: List of reference texts
        device: Device to run on
        
    Returns:
        Dictionary with WER and CER metrics
    """
    # Set decoder prompt for English transcription
    model.config.forced_decoder_ids = (
        processor.tokenizer.get_decoder_prompt_ids(task="transcribe", language="english")
    )
    
    predictions = []
    references = []
    norm_predictions = []
    norm_references = []
    
    print(f"Evaluating model on {len(audio_paths)} samples...")
    
    # Process dataset item by item
    for i in tqdm(range(len(audio_paths)), desc="Evaluating"):
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
        norm_predictions.append(normalize_text(pred_text))
        norm_references.append(normalize_text(ref_text))
    
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

def main():
    """Main evaluation function."""
    # Load original model and processor
    print("Loading Whisper model...")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    
    # Move model to GPU
    model = model.to("cuda:0")
    print("Model moved to GPU")
    
    # Load evaluation dataset from custom files
    print("Loading evaluation dataset...")
    with open("custom_data/LibriSpeech/test-clean/audio_paths", "r") as f:
        audio_paths = [line.strip().split(" ", 1)[1] for line in f.readlines()]
    with open("custom_data/LibriSpeech/test-clean/text", "r") as f:
        texts = [line.strip().split(" ", 1)[1] for line in f.readlines()]
    
    # Take first 100 samples
    audio_paths = audio_paths[:100]
    texts = texts[:100]
    
    print(f"Loaded {len(audio_paths)} samples for evaluation")

    # Define sparsity levels to test
    sparsities = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    
    # Results dictionary
    results = {}
    
    # Audio path for pruning
    audio_path = "/datasets/speech/LibriSpeech/dev-clean/3081/166546/3081-166546-0000.flac"
    
    print(f"Evaluating {len(sparsities)} sparsity levels...")
    
    for sparsity in sparsities:
        print(f"\n{'='*60}")
        print(f"Evaluating sparsity: {sparsity:.1%}")
        print(f"{'='*60}")
        
        # Prune model
        if sparsity == 0.0:
            pruned_model = model
        else:
            pruned_model = utility_obs_prune(
                model=model,
                processor=processor,
                audio_path=audio_path,
                sparsity=sparsity,
                debug=False  
            )
        
        # Evaluate pruned model
        metrics = evaluate_model(
            model=pruned_model,
            processor=processor,
            audio_paths=audio_paths,
            texts=texts,
            device=0
        )
        
        # Store results
        results[sparsity] = metrics
        
        print(f"Results for {sparsity:.1%} sparsity:")
        print(f"  WER: {metrics['wer']:.2f}%")
        print(f"  CER: {metrics['cer']:.2f}%")
        print(f"  Normalized WER: {metrics['normalized_wer']:.2f}%")
        print(f"  Normalized CER: {metrics['normalized_cer']:.2f}%")
        print("\n\n\n\n")
    
        # Clean up GPU memory
        del pruned_model
        torch.cuda.empty_cache()

    # Save results to JSON
    output_file = "pruned_model_evaluation_results.json"
    print(f"\n{'='*60}")
    print(f"Saving results to {output_file}")
    print(f"{'='*60}")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Evaluation completed!")
    print(f"Results saved to: {output_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Sparsity':<10} {'WER':<8} {'CER':<8} {'Norm WER':<10} {'Norm CER':<10}")
    print("-" * 60)
    
    for sparsity, metrics in results.items():
        if "error" not in metrics:
            print(f"{sparsity:<10.1%} {metrics['wer']:<8.2f} {metrics['cer']:<8.2f} {metrics['normalized_wer']:<10.2f} {metrics['normalized_cer']:<10.2f}")
        else:
            print(f"{sparsity:<10.1%} {'ERROR':<8} {'ERROR':<8} {'ERROR':<10} {'ERROR':<10}")

if __name__ == "__main__":
    main()
