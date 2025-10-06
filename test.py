from transformers import WhisperForConditionalGeneration, WhisperProcessor
from utils.obs import utility_obs_prune

if __name__ == "__main__":   
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    
    utility_obs_prune(model, processor, audio_path="/datasets/speech/LibriSpeech/dev-clean/3081/166546/3081-166546-0000.flac", sparsity=0.5, debug=True)