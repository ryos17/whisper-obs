ngpu=1  # number of GPUs to perform distributed training on.

torchrun --nproc_per_node=${ngpu} train/fine-tune_on_hf_dataset.py \
--model_name openai/whisper-tiny \
--sampling_rate 16000 \
--num_proc 16 \
--train_strategy epoch \
--learning_rate 3e-3 \
--warmup 10 \
--train_batchsize 128 \
--eval_batchsize 64 \
--num_epochs 10 \
--resume_from_ckpt None \
--output_dir op_dir_steps \
--train_datasets ahazeemi/librispeech10h ahazeemi/librispeech10h \
--train_dataset_splits train.10 validation \
--train_dataset_text_columns text text \
--eval_datasets ahazeemi/librispeech10h \
--eval_dataset_splits test \
--eval_dataset_text_columns text