ngpu=1
MASTER_PORT=29501
CUDA_VISIBLE_DEVICES=1

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} torchrun --nproc_per_node=${ngpu} --master_port=${MASTER_PORT} train/fine-tune_on_custom_dataset.py \
--model_name openai/whisper-tiny.en \
--sampling_rate 16000 \
--num_cpu_workers 16 \
--train_strategy epoch \
--learning_rate 3.75e-5 \
--warmup 100 \
--train_batchsize 128 \
--eval_batchsize 64 \
--num_epochs 10 \
--resume_from_ckpt None \
--output_dir output/$(basename "$0" .sh) \
--train_datasets librispeech/train-clean-100 \
--eval_datasets librispeech/dev-clean \