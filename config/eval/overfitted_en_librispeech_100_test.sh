python evaluate/evaluate_on_custom_dataset.py \
--is_public_repo False \
--ckpt_dir "output/en_librispeech_100_test/checkpoint-2230" \
--temp_ckpt_folder temp/$(basename "$0" .sh) \
--eval_datasets librispeech/test-clean \
--device 1 \
--batch_size 64 \
--output_dir output_eval/$(basename "$0" .sh)