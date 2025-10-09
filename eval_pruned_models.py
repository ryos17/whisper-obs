import json
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from utils.obs import utility_obs_prune, utility_obs_evaluate
from utils.imp import utility_imp_prune, utility_imp_evaluate
import torch

def main():
    # Load original model and processor
    print("Loading Whisper model...")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

    # Define sparsity levels to test
    sparsities = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    
    # Results dictionary
    results = {}
    
    # Audio path for pruning
    audio_path = "/datasets/speech/LibriSpeech/dev-clean/3081/166546/3081-166546-0000.flac"
    
    print(f"Evaluating {len(sparsities)} sparsity levels...")
    
    for sparsity in sparsities:
        print("=" * 60)
        print(f"{f'Sparsity: {sparsity:.1%}':^60}")
        
        # Prune model
        if sparsity == 0.0:
            pruned_model = model
        else:
            pruned_model = utility_imp_prune(
                model=model,
                processor=processor,
                audio_path=audio_path,
                sparsity=sparsity,
                debug=False  
            )
        
        # Evaluate pruned model
        metrics = utility_imp_evaluate(
            model=pruned_model,
            processor=processor,
            num_samples=100
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
    output_file = "pruned_model_evaluation_results.json"
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
