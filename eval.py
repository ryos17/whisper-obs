import json
import argparse
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from utils.obs import utility_obs_prune, utility_obs_evaluate
from utils.mp import utility_mp_prune, utility_mp_evaluate
import torch
import copy

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pruned Whisper models at different sparsity levels")
    parser.add_argument("--model", type=str, default="openai/whisper-tiny", 
                        help="Whisper model name or path (default: openai/whisper-tiny)")
    parser.add_argument("--output", type=str, default="obs_results.json",
                        help="Path to save results JSON file (default: obs_results.json)")
    parser.add_argument("--audio", type=str, 
                        default="/datasets/speech/LibriSpeech/dev-clean/3081/166546/3081-166546-0000.flac",
                        help="Path to audio file for pruning")
    parser.add_argument("--method", type=str, default="obs", choices=["obs", "mp_local", "mp_global"],
                        help="Pruning method to use (default: obs)")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples for evaluation (default: 100)")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU device ID to use (default: 0)")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load original model and processor
    print(f"Loading Whisper model: {args.model}...")
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    processor = WhisperProcessor.from_pretrained(args.model)

    # Define sparsity levels to test
    sparsities = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    
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
                    device=args.device
                )
            elif args.method == "mp_local":
                pruned_model = utility_mp_prune(
                    model=model,
                    processor=processor,
                    audio_path=args.audio,
                    sparsity=sparsity,
                    prune_method="local",
                    device=args.device
                )
            elif args.method == "mp_global":
                pruned_model = utility_mp_prune(
                    model=model,
                    processor=processor,
                    audio_path=args.audio,
                    sparsity=sparsity,
                    prune_method="global",
                    device=args.device
                )
        
        # Evaluate pruned model
        if args.method == "obs":
            metrics = utility_obs_evaluate(
                model=pruned_model,
                processor=processor,
                num_samples=args.num_samples,
                device=args.device,
            )
        else:  
            metrics = utility_mp_evaluate(
                model=pruned_model,
                processor=processor,
                num_samples=args.num_samples,
                device=args.device,
            )
        
        # Clean up GPU memory
        del pruned_model
        torch.cuda.empty_cache()
        
        # Store results
        results[sparsity] = metrics

        print(f"{'-'*60}")
        print(f"{'WER:':<25} {metrics['wer']:.2f}%")
        print(f"{'CER:':<25} {metrics['cer']:.2f}%")
        print(f"{'Normalized WER:':<25} {metrics['normalized_wer']:.2f}%")
        print(f"{'Normalized CER:':<25} {metrics['normalized_cer']:.2f}%")

    # Save results to JSON
    output_file = args.output
    print(f"\n{'='*60}")
    print(f"Saving results to {output_file}")
    print("=" * 60)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Evaluation completed!")
    print(f"Results saved to: {output_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Sparsity':<10} {'WER':<8} {'CER':<8} {'Norm WER':<10} {'Norm CER':<10}")
    print("-" * 60)
    
    for sparsity, metrics in results.items():
        if "error" not in metrics:
            print(f"{sparsity:<10.1%} {metrics['wer']:<8.2f} {metrics['cer']:<8.2f} {metrics['normalized_wer']:<10.2f} {metrics['normalized_cer']:<10.2f}")
        else:
            print(f"{sparsity:<10.1%} {'ERROR':<8} {'ERROR':<8} {'ERROR':<10} {'ERROR':<10}")

if __name__ == "__main__":
    main()
