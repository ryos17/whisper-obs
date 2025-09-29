import os
import argparse
from pathlib import Path

def process_librispeech_directory(root_dir, output_dir):
    """
    Process a single LibriSpeech subset directory and create audio_paths and text files.
    
    Args:
        root_dir: Single LibriSpeech subset directory (e.g., dev-clean, train-clean-100)
        output_dir: Directory to save output files
    
    Returns:
        Number of processed files
    """
    root_path = Path(root_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    audio_paths = []
    transcriptions = []
    
    print(f"Processing directory: {root_path}")
    
    # Walk through all subdirectories
    for speaker_dir in root_path.iterdir():
        if not speaker_dir.is_dir():
            continue
            
        print(f"  Processing speaker: {speaker_dir.name}")
        
        # Process each chapter directory
        for chapter_dir in speaker_dir.iterdir():
            if not chapter_dir.is_dir():
                continue
                
            # Look for .trans.txt file
            trans_file = chapter_dir / f"{speaker_dir.name}-{chapter_dir.name}.trans.txt"
            
            if not trans_file.exists():
                print(f"    Warning: No transcript file found for {chapter_dir}")
                continue
            
            # Read transcriptions
            with open(trans_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Split into ID and transcription
                    parts = line.split(' ', 1)
                    if len(parts) != 2:
                        continue
                    
                    unique_id = parts[0]
                    transcription = parts[1]
                    
                    # Find corresponding audio file
                    audio_file = chapter_dir / f"{unique_id}.flac"
                    if not audio_file.exists():
                        print(f"    Warning: Audio file not found for {unique_id}")
                        continue
                    
                    # Add to lists
                    audio_paths.append(f"{unique_id} {audio_file.absolute()}")
                    transcriptions.append(f"{unique_id} {transcription}")
    
    # Write audio paths file
    audio_paths_file = output_path / "audio_paths"
    with open(audio_paths_file, 'w', encoding='utf-8') as f:
        for line in audio_paths:
            f.write(line + '\n')
    
    # Write transcriptions file
    text_file = output_path / "text"
    with open(text_file, 'w', encoding='utf-8') as f:
        for line in transcriptions:
            f.write(line + '\n')
    
    print(f"  Processed {len(audio_paths)} audio files")
    print(f"  Audio paths saved to: {audio_paths_file}")
    print(f"  Transcriptions saved to: {text_file}")
    
    return len(audio_paths)

def process_librispeech_dataset(parent_dir, custom_data_dir="custom_data"):
    """
    Process entire LibriSpeech dataset and create files for each subset.
    
    Args:
        parent_dir: Parent LibriSpeech directory (e.g., /datasets/speech/LibriSpeech)
        custom_data_dir: Base directory for custom data (default: "custom_data")
    
    Returns:
        Dictionary with subset names and file counts
    """
    parent_path = Path(parent_dir)
    dataset_name = parent_path.name  # Use basename of parent directory
    custom_data_path = Path(custom_data_dir) / dataset_name
    
    # Create base custom data directory
    custom_data_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    print(f"Processing dataset from: {parent_path}")
    print(f"Dataset name: {dataset_name}")
    print(f"Output base directory: {custom_data_path}")
    print("=" * 60)
    
    # Find all subset directories
    subset_dirs = [d for d in parent_path.iterdir() if d.is_dir()]
    
    if not subset_dirs:
        print(f"Error: No subdirectories found in {parent_path}")
        return results
    
    print(f"Found {len(subset_dirs)} subsets: {[d.name for d in subset_dirs]}")
    print()
    
    # Process each subset
    for subset_dir in subset_dirs:
        subset_name = subset_dir.name
        output_dir = custom_data_path / subset_name
        
        print(f"Processing subset: {subset_name}")
        print("-" * 40)
        
        try:
            count = process_librispeech_directory(subset_dir, output_dir)
            results[subset_name] = count
            print(f"✓ Completed {subset_name}: {count} files")
        except Exception as e:
            print(f"✗ Error processing {subset_name}: {e}")
            results[subset_name] = 0
        
        print()
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Process LibriSpeech parent directory and create files for each subset')
    parser.add_argument('parent_dir', help='Parent LibriSpeech directory (e.g., /datasets/speech/LibriSpeech)')
    parser.add_argument('--custom_data_dir', '-c', default='custom_data',
                       help='Base directory for custom data (default: custom_data)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.parent_dir):
        print(f"Error: Directory {args.parent_dir} does not exist")
        return 1
    
    try:
        results = process_librispeech_dataset(args.parent_dir, args.custom_data_dir)
        
        print("=" * 60)
        print("SUMMARY:")
        total_files = 0
        for subset, count in results.items():
            print(f"  {subset}: {count} files")
            total_files += count
        
        print(f"\nTotal files processed: {total_files}")
        dataset_name = Path(args.parent_dir).name
        print(f"Output directory: {Path(args.custom_data_dir) / dataset_name}")
        
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())