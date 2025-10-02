ngpu=1
MASTER_PORT=29500
CUDA_VISIBLE_DEVICES=0

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} torchrun --nproc_per_node=${ngpu} --master_port=${MASTER_PORT} train/fine-tune_on_custom_dataset.py \
--model_name openai/whisper-tiny \
--language english \
--sampling_rate 16000 \
--num_cpu_workers 8 \
--train_strategy steps \
--learning_rate 3.75e-5 \
--warmup 400 \
--train_batchsize 128 \
--eval_batchsize 64 \
--num_steps 8000 \
--eval_save_steps 400 \
--resume_from_ckpt None \
--output_dir output/$(basename "$0" .sh) \
--train_datasets librispeech/train-clean-360 \
--eval_datasets librispeech/dev-clean \