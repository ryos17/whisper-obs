python3 evaluate/evaluate_on_custom_dataset.py \
--is_public_repo True \
--hf_model openai/whisper-tiny.en \
--eval_datasets librispeech/test-clean \
--device 1 \
--batch_size 64 \
--output_dir output_eval/$(basename "$0" .sh)