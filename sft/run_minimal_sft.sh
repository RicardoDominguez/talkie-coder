#!/bin/bash
# Minimum-data SFT sweep launcher. Same hyperparams as run_coder_sft_12h_v2_lr2e5.sh
# (lr=2e-5, max_grad_norm=30, warmup_ratio=0.03, ...) but on the talkie-1930-13b-it
# (instruct) model and with --subsample to cap the unique-data budget. max_steps
# stays the lever for total optimizer compute; the trainer auto-loops to fill it.
#
# Usage:
#   run_minimal_sft.sh <max_steps> <subsample> <run_name> <output_dir>
#
# Example (Run 3, s20_e3):
#   run_minimal_sft.sh 20 0.0047 talkie-1930-it-coder-s20-e3 \
#     /fast/rolmedo/swe-models/talkie-1930-it-coder-s20-e3
set -u

MAX_STEPS="${1:?max_steps required}"
SUBSAMPLE="${2:?subsample required}"
RUN_NAME="${3:?run_name required}"
OUTPUT_DIR="${4:?output_dir required}"

source /home/rolmedo/tflatest/bin/activate
export HF_HOME=/tmp HOME=/lustre/home/rolmedo WANDB_DIR=/tmp HF_TOKEN_PATH=~/.cache/huggingface/token
export WANDB_PROJECT=${WANDB_PROJECT:-swe-tune}
module load cuda/12.4 2>/dev/null

cd /home/rolmedo/talkie/sft

accelerate launch --config_file accelerate_config_talkie.yaml ft_trl.py \
  --model /fast/rolmedo/models/talkie-1930-13b-it \
  --train_dataset_dir /fast/rolmedo/swesmith/datasets/talkie-1930-mini-coder-vmax3-64k \
  --subsample "$SUBSAMPLE" \
  --max_length 65536 \
  --run_name "$RUN_NAME" \
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
  --max_steps "$MAX_STEPS" --save_strategy no \
  --ckpt_every_steps 0 \
  --output_dir "$OUTPUT_DIR"
