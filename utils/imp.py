import torch
import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
from utils.mp import utility_mp_prune
from datasets import load_from_disk, Audio
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


def utility_imp_prune(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    sparsities: List[float],
    device: int = 0,
    prune_method: str = "global",
    train_batch_size: int = 8,
    warmup_steps: int = 400,
    num_cpu_workers: int = 32,
    epochs: int = 2,
    learning_rate: float = 3.75e-5,
    debug: bool = False,
):
    """
    Prune a Whisper model using the Iterative Magnitude Pruning algorithm
    with fine-tuning between pruning steps.
    
    Args:
        model: Whisper model to prune
        processor: Whisper processor
        sparsities: List of sparsity levels to prune iteratively
        device: Device to run on (default: 0)
        prune_method: Method to use for pruning (default: "global")
        train_batch_size: Batch size for fine-tuning (default: 8)
        warmup_steps: Warmup steps for fine-tuning (default: 400)
        num_cpu_workers: Number of CPU workers for data loading (default: 32)
        epochs: Number of fine-tuning epochs after each pruning step (default: 2)
        learning_rate: Learning rate for fine-tuning (default: 3.75e-5)
        debug: Whether to print debug information (default: False)
        
    Returns:
        Pruned model
    """
    # Hardcoded path for fine-tuning
    finetune_dataset_path = "librispeech/train-clean-100"

    # Load the fine-tuning dataset
    train_dataset = load_from_disk(finetune_dataset_path)
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    whisper_norm = BasicTextNormalizer()

    # Processing and prepare_dataset
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]

        # Use "sentence" column for transcripts 
        transcription = batch["sentence"] if "sentence" in batch else (batch["text"] if "text" in batch else "")
        transcription = whisper_norm(transcription)
        
        batch["labels"] = processor.tokenizer(transcription).input_ids
        return batch

    train_dataset = train_dataset.map(prepare_dataset, num_proc=num_cpu_workers)
    
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="output/test",
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="no",
        save_strategy="no",
        num_train_epochs=epochs,
        logging_steps=500,
        report_to=["tensorboard"],
        optim="adamw_bnb_8bit",
        dataloader_num_workers=num_cpu_workers,
    )

        
    # Create a deep copy of the model to prune
    pruned_model = copy.deepcopy(model)

    # Process each sparsity level iteratively
    for sparsity in sparsities:    
        # Apply MP pruning for this stage
        pruned_model = utility_mp_prune(
            pruned_model,
            processor,
            sparsity,
            device=device,
            debug=debug,
            prune_method=prune_method
        )

        # Fine-tune the pruned model
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=pruned_model,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=processor.feature_extractor,
        )

        trainer.train()
    
    # Return the pruned model
    return pruned_model
