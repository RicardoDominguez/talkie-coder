source /home/rolmedo/tflatest/bin/activate

set -x

export HF_HOME=/tmp
export HF_TOKEN_PATH=~/.cache/huggingface/token
export HOME=/lustre/home/rolmedo/

export WANDB_DIR="/tmp/"

export VLLM_ATTENTION_BACKEND=XFORMERS

module load cuda/12.4

$@ 