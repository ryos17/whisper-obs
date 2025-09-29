ngpu=1  
MASTER_PORT=29500
CUDA_VISIBLE_DEVICES=0

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} torchrun --nproc_per_node=${ngpu} --master_port=${MASTER_PORT} train/fine-tune_on_hf_dataset.py \
--model_name openai/whisper-tiny \
--sampling_rate 16000 \
--num_proc 16 \
--train_strategy epoch \
--learning_rate 3.75e-5 \
--warmup 25 \
--train_batchsize 128 \
--eval_batchsize 64 \
--num_epochs 10 \
--resume_from_ckpt None \
--output_dir output/hf_test \
--train_datasets ahazeemi/librispeech10h ahazeemi/librispeech10h \
--train_dataset_splits train.10 validation \
--train_dataset_text_columns text text \
--eval_datasets ahazeemi/librispeech10h \
--eval_dataset_splits test \
--eval_dataset_text_columns text