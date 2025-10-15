import json
import argparse
import torch
import copy
from tqdm import tqdm
import evaluate
import torchaudio
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from utils.obs import utility_obs_prune
from utils.mp import utility_mp_prune
from utils.iobs import utility_iobs_prune
from utils.imp import utility_imp_prune

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pruned Whisper models at different sparsity levels")
    parser.add_argument("--model", type=str, default="openai/whisper-tiny", 
                        help="Whisper model name or path (default: openai/whisper-tiny)")
    parser.add_argument("--output", type=str, default="obs_results.json",
                        help="Path to save results JSON file (default: obs_results.json)")
    parser.add_argument("--audio", type=str, 
                        default="/datasets/speech/LibriSpeech/dev-clean/3081/166546/3081-166546-0000.flac",
                        help="Path to audio file for pruning")
    parser.add_argument("--method", type=str, default="obs", choices=["obs", "obs_finetune", "iobs", "mp_local", "mp_global", "mp_finetune_local", "mp_finetune_global", "imp_local", "imp_global"],
                        help="Pruning method to use (default: obs)")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples for evaluation (default: 100)")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU device ID to use (default: 0)")
    parser.add_argument("--sparsities", type=str, default="0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0",
                        help="Sparsities to evaluate (default: 0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0)")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Whether to print debug information (default: False)")
    return parser.parse_args()


def count_nonzero_params(model):
    nonzero_prunable = 0
    total_prunable = 0
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and isinstance(module, torch.nn.Linear):
            nonzero_prunable += (module.weight != 0).sum().item()
            total_prunable += module.weight.numel()
    return nonzero_prunable, total_prunable


def frange(start, stop, step):
    if stop < start:
        return [stop]
    vals = []
    v = start
    while v <= stop:  
        vals.append(round(v, 2))
        v += step
    return vals


def evaluate_pruned_model(model, processor, num_samples=None, device=0, debug=False):
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

            if debug:   
                print(f"{'Predicted text:':<20} {pred_text}")
                print(f"{'Reference text:':<20} {ref_text}")
    
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

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load original model and processor
    print(f"Loading Whisper model: {args.model}...")
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    processor = WhisperProcessor.from_pretrained(args.model)

    # Define sparsity levels to test
    sparsities = [float(s) for s in args.sparsities.split(",")]
    
    # Results dictionary
    results = {}
    
    print(f"Evaluating {len(sparsities)} sparsity levels using {args.method} method...")
    
    for sparsity in sparsities:
        print("=" * 60)
        print(f"{f'Sparsity: {sparsity:.1%}':^60}")
        
        # Prune model
        if sparsity == 0.0:
            pruned_model = copy.deepcopy(model)
        else:
            if args.method == "obs":
                pruned_model = utility_obs_prune(
                    model=model,
                    processor=processor,
                    audio_path=args.audio,
                    sparsity=sparsity,
                    device=args.device,
                    debug=args.debug
                )
            elif args.method == "obs_finetune":
                pruned_model = utility_iobs_prune(
                    model=model,
                    processor=processor,
                    audio_path=args.audio,
                    sparsities=[sparsity],
                    device=args.device,
                    debug=args.debug
                )
            elif args.method == "iobs":
                pruned_model = utility_iobs_prune(
                    model=model,
                    processor=processor,
                    audio_path=args.audio,
                    sparsities=frange(0.4, sparsity, 0.1),
                    device=args.device,
                    debug=args.debug
                )
            elif args.method == "mp_local":
                pruned_model = utility_mp_prune(
                    model=model,
                    processor=processor,
                    audio_path=args.audio,
                    sparsity=sparsity,
                    prune_method="local",
                    device=args.device,
                    debug=args.debug
                )
            elif args.method == "mp_global":
                pruned_model = utility_mp_prune(
                    model=model,
                    processor=processor,
                    audio_path=args.audio,
                    sparsity=sparsity,
                    prune_method="global",
                    device=args.device,
                    debug=args.debug
                )
            elif args.method == "mp_finetune_local":
                pruned_model = utility_imp_prune(
                    model=model,
                    processor=processor,
                    audio_path=args.audio,
                    sparsities=[sparsity],
                    prune_method="local",
                    device=args.device,
                    debug=args.debug,
                )
            elif args.method == "mp_finetune_global":
                pruned_model = utility_imp_prune(
                    model=model,
                    processor=processor,
                    audio_path=args.audio,
                    sparsities=[sparsity],
                    prune_method="global",
                    device=args.device,
                    debug=args.debug
                )
            elif args.method == "imp_local":
                pruned_model = utility_imp_prune(
                    model=model,
                    processor=processor,
                    audio_path=args.audio,
                    sparsities=frange(0.3, sparsity, 0.05),
                    prune_method="local",
                    device=args.device,
                    debug=args.debug,
                )
            elif args.method == "imp_global":
                pruned_model = utility_imp_prune(
                    model=model,
                    processor=processor,
                    audio_path=args.audio,
                    sparsities=frange(0.3, sparsity, 0.05),
                    prune_method="global",
                    device=args.device,
                    debug=args.debug
                )
            else:
                raise ValueError(f"Invalid method: {args.method}")

        metrics = evaluate_pruned_model(
            model=pruned_model,
            processor=processor,
            num_samples=args.num_samples,
            device=args.device,
            debug=args.debug,
        )

        # Actual sparsity
        pruned_nonzero, total_params = count_nonzero_params(pruned_model)
        actual_sparsity = 1 - (pruned_nonzero / total_params) if total_params > 0 else 0
        metrics['actual_sparsity'] = actual_sparsity

        # Clean up GPU memory
        del pruned_model
        torch.cuda.empty_cache()
        
        # Store results
        results[sparsity] = metrics

        print(f"{'-'*60}")
        print(f"{'Actual sparsity:':<30} {actual_sparsity:.2%}")
        print(f"{'WER:':<30} {metrics['wer']:.2f}%")
        print(f"{'CER:':<30} {metrics['cer']:.2f}%")
        print(f"{'Normalized WER:':<30} {metrics['normalized_wer']:.2f}%")
        print(f"{'Normalized CER:':<30} {metrics['normalized_cer']:.2f}%")

    # Save results to JSON
    output_file = args.output
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("=" * 60)
    print("Evaluation completed!")
    print(f"Results saved to: {output_file}")
    
    # Print summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Sparsity':<10} {'WER':<8} {'CER':<8} {'Norm WER':<10} {'Norm CER':<10} {'Actual Sparsity':<10}")
    print("-" * 60)
    
    for sparsity, metrics in results.items():
        print(f"{sparsity:<10.1%} {metrics['wer']:<8.2f} {metrics['cer']:<8.2f} {metrics['normalized_wer']:<10.2f} {metrics['normalized_cer']:<10.2f} {metrics['actual_sparsity']:<10.2%}")

if __name__ == "__main__":
    main()
