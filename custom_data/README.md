# LibriSpeech Data Preparation

This script processes the LibriSpeech dataset and creates the required `audio_paths` and `text` files for each subset.

## Usage

```bash
# Process entire LibriSpeech dataset
python librispeech_prep.py /datasets/speech/LibriSpeech

# With custom dataset name
python librispeech_prep.py /datasets/speech/LibriSpeech --dataset_name "MyLibriSpeech"

# With custom output directory
python librispeech_prep.py /datasets/speech/LibriSpeech --custom_data_dir "my_data"
```

## Output Structure

The script will create the following structure:

```
custom_data/
└── LibriSpeech/
    ├── dev-clean/
    │   ├── audio_paths
    │   └── text
    ├── dev-other/
    │   ├── audio_paths
    │   └── text
    ├── test-clean/
    │   ├── audio_paths
    │   └── text
    ├── test-other/
    │   ├── audio_paths
    │   └── text
    ├── train-clean-100/
    │   ├── audio_paths
    │   └── text
    ├── train-clean-360/
    │   ├── audio_paths
    │   └── text
    └── train-other-500/
        ├── audio_paths
        └── text
```

## File Format

Each `audio_paths` file contains:
```
<unique-id> <absolute path to the audio file>
```

Each `text` file contains:
```
<unique-id> <Transcription (ground truth) corresponding to the audio file>
```

## Example

```bash
# Run the script
cd /home/ryota/whisper-finetune
python custom_data/librispeech_prep.py /datasets/speech/LibriSpeech
```

This will process all LibriSpeech subsets and create the required files in `custom_data/LibriSpeech/`.
