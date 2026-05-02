#!/bin/bash
# Sibling of run_swe_sft_12h_v2_lr2e5.sh: same recipe (lr=2e-5,
# max_grad_norm=30, 2016 steps, ckpt every 200) but base model is the
# chat-token-reinit'd talkie-web base and dataset is the web-tokenized
# swe-100k (no per-instance cap, no verified filter).
set -u
source /home/rolmedo/tflatest/bin/activate
export HF_HOME=/tmp HOME=/lustre/home/rolmedo WANDB_DIR=/tmp HF_TOKEN_PATH=~/.cache/huggingface/token
export WANDB_PROJECT=${WANDB_PROJECT:-swe-tune}
module load cuda/12.4 2>/dev/null

accelerate launch --config_file accelerate_config_talkie.yaml ft_trl.py \
  --model /fast/rolmedo/models/talkie-web-13b-base-reinit \
  --train_dataset_dir /fast/rolmedo/swesmith/datasets/talkie-web-swe-100k-64k \
  --max_length 65536 \
  --run_name talkie-web-base-swe-12h-v2-lr2e5 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --packing --packing_strategy bfd --padding_free \
  --dataset_num_proc 32 \
  --gradient_checkpointing --use_liger_kernel \
  --trust_remote_code 1 --attn_implementation none \
  --bf16 1 --optim adamw_torch_fused \
  --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-8 \
  --weight_decay 0.1 --max_grad_norm 30 \
  --lr_scheduler_type cosine_with_min_lr --warmup_ratio 0.03 \
  --completion_only_loss 1 --report_to wandb \
  --average_tokens_across_devices False \
  --logging_steps 1 \
  --max_steps 2016 --save_strategy no \
  --ckpt_every_steps 200 \
  --output_dir /fast/rolmedo/swe-models/talkie-web-13b-base-swe-12h-v2-lr2e5
