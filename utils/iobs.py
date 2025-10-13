import torch
import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
from utils.obs import utility_obs_prune
from datasets import load_from_disk, Audio
import evaluate
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


def utility_iobs_prune(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio_path: str,
    sparsities: List[float],
    batch_size: int = 128,
    device: int = 0,
    alpha: float = 0.03,
    train_batch_size: int = 8,
    eval_batch_size: int = 8,
    num_cpu_workers: int = 32,
    epochs: int = 2,
    learning_rate: float = 3.75e-5,
    debug: bool = False,
):
    """
    Prune a Whisper model using the Iterative Optimal Brain Surgeon algorithm
    with fine-tuning between pruning steps.
    
    Args:
        model: Whisper model to prune
        processor: Whisper processor
        audio_path: Path to the calibration audio file
        sparsities: Single sparsity level or list of sparsity levels to prune iteratively
        batch_size: Batch size for pruning (default: 128)
        device: Device to run on (default: 0)
        alpha: Hyperparameter controlling the range of sparsity for mixed pruning (default: 0.03)
        train_batch_size: Batch size for fine-tuning (default: 8)
        eval_batch_size: Batch size for evaluation during fine-tuning (default: 8)
        num_cpu_workers: Number of CPU workers for data loading (default: 32)
        epochs: Number of fine-tuning epochs after each pruning step (default: 2)
        learning_rate: Learning rate for fine-tuning (default: 3.75e-5)
        debug: Whether to print debug information (default: False)
        
    Returns:
        Pruned model
    """
    # Hardcoded paths for fine-tuning
    finetune_dataset_path = "librispeech/train-clean-100"
    eval_dataset_path = "librispeech/dev-clean"

    # Load the fine-tuning datasets
    train_dataset = load_from_disk(finetune_dataset_path)
    eval_dataset = load_from_disk(eval_dataset_path)
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=16000))
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
    eval_dataset = eval_dataset.map(prepare_dataset, num_proc=num_cpu_workers)
    
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
        warmup_steps=10,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=epochs,
        per_device_eval_batch_size=eval_batch_size,
        predict_with_generate=True,
        generation_max_length=100,
        logging_steps=500,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        optim="adamw_bnb_8bit",
        dataloader_num_workers=num_cpu_workers,
    )

    # Define metrics
    metric = evaluate.load("wer")
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}
        
    # Create a deep copy of the model to prune
    pruned_model = copy.deepcopy(model)
    
    # Process each sparsity level iteratively
    for sparsity in sparsities:    
        # Apply OBS pruning for this stage
        pruned_model = utility_obs_prune(
            pruned_model,
            processor,
            audio_path,
            sparsity,
            batch_size=batch_size,
            device=device,
            debug=debug,
            alpha=alpha
        )

        # Fine-tune the pruned model
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=pruned_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=processor.feature_extractor,
        )

        trainer.train()
    
    # Return the pruned model
    return pruned_model
