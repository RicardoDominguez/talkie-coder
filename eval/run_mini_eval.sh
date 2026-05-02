#!/bin/bash
# Run mini-swe-agent batch on a SLICE of SWE-bench (default 0:100) against the
# v2-ckpt400 talkie checkpoint. Same shape as run_mini_smoke.sh; bumped workers
# for parallelism.
set -uo pipefail

# HTCondor strips the user's PATH; restore the standard search path.
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export USER="${USER:-$(id -un)}"
export HOME="${HOME:-/lustre/home/$USER}"
export PATH="$HOME/bin:$PATH"
export HF_HOME=/tmp HF_DATASETS_CACHE=/tmp HF_MODELS_CACHE=/tmp
export TRITON_CACHE_DIR=/tmp/triton TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor
export VLLM_CACHE_ROOT=/tmp/vllm_cache

MODEL_DIR="${MODEL_DIR:-/fast/rolmedo/swe-models/talkie-web-12h-v2-ckpt400-vllm}"
OUTPUT_DIR="${OUTPUT_DIR:-/fast/rolmedo/swesmith/talkie-web-v2-ckpt400-mini-100}"
SLICE="${SLICE:-0:100}"
SUBSET="${SUBSET:-verified_cluster}"
SPLIT="${SPLIT:-test}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-/lustre/home/rolmedo/talkie-eval/swe-evals/localconfig_qwen3_train_aligned.yaml}"
NUM_WORKERS="${NUM_WORKERS:-3}"
HOSTNAME="$(hostname)"

# Pick a free port for vLLM
PORT=$(env HOSTNAME="$HOSTNAME" python3 - <<'PY'
import os, socket
host = os.environ.get('HOSTNAME') or socket.gethostname()
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind((host, 0))
except OSError:
    s.bind(('', 0))
print(s.getsockname()[1])
s.close()
PY
)
echo "[run_mini_eval] HOSTNAME=$HOSTNAME PORT=$PORT MODEL_DIR=$MODEL_DIR SLICE=$SLICE WORKERS=$NUM_WORKERS"

# Background /tmp + cgroup memory diagnostics — every 60s, dump top users
# of /tmp and the actual cgroup memory.usage_in_bytes vs limit.
(
  while true; do
    ts=$(date +%T)
    echo "[diag $ts] /tmp top: $(du -sBM /tmp/* 2>/dev/null | sort -rn | head -8 | tr '\n' ' ')"
    cg=$(grep memory /proc/self/cgroup | cut -d: -f3)
    if [ -n "$cg" ] && [ -d "/sys/fs/cgroup/memory$cg" ]; then
      u=$(cat /sys/fs/cgroup/memory$cg/memory.usage_in_bytes 2>/dev/null)
      l=$(cat /sys/fs/cgroup/memory$cg/memory.limit_in_bytes 2>/dev/null)
      [ -n "$u$l" ] && echo "[diag $ts] cgroup: usage=$((u/1024/1024))MB limit=$((l/1024/1024))MB"
    fi
    sleep 60
  done
) &
DIAG_PID=$!

# 1. Start rootless docker
bash /lustre/home/rolmedo/talkie-eval/swe-evals/docker_setup.sh
sleep 3

# 2. Spawn vLLM (background)
module load cuda/12.4 2>/dev/null
source /home/rolmedo/vllm019/bin/activate
vllm serve "$MODEL_DIR" \
  --model-impl transformers --trust-remote-code \
  --dtype bfloat16 --max-model-len 32768 --tensor-parallel-size 1 \
  --host "$HOSTNAME" --port "$PORT" \
  --generation-config vllm > /tmp/vllm.log 2>&1 &
VLLM_PID=$!

# 3. Wait for vLLM ready
echo "[run_mini_eval] waiting for vLLM at $HOSTNAME:$PORT"
for i in $(seq 1 600); do
  if no_proxy="$HOSTNAME" curl -s "http://$HOSTNAME:$PORT/v1/models" 2>/dev/null | grep -q '"id"'; then
    echo "[run_mini_eval] vLLM ready"
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[run_mini_eval] vLLM exited unexpectedly; tail of /tmp/vllm.log:"
    tail -40 /tmp/vllm.log
    exit 1
  fi
  sleep 2
done

# 4. Build per-job model config
DEST_CONFIG="/tmp/modelconfig_eval.yaml"
SRC="$CONFIG_TEMPLATE" DEST="$DEST_CONFIG" TEMPERATURE=0.7 MAX_TOKENS=4096 \
  /home/rolmedo/swe-evals/transfer_config.sh "$HOSTNAME" "$PORT" "$MODEL_DIR"

mkdir -p "$OUTPUT_DIR"

# 5. Run mini-extra swebench batch
source /home/rolmedo/miniswa/bin/activate

# Build args: prefer INSTANCE_IDS (subset eval) over SLICE when both set.
EXTRA_ARGS=()
if [ -n "${INSTANCE_IDS:-}" ]; then
  EXTRA_ARGS+=(--instance-ids "$INSTANCE_IDS")
elif [ -n "${SLICE:-}" ]; then
  EXTRA_ARGS+=(--slice "$SLICE")
fi

DOCKER_HOST=unix:///tmp/docker.sock \
no_proxy="$HOSTNAME" NO_PROXY="$HOSTNAME" \
mini-extra swebench \
  --subset "$SUBSET" \
  --split "$SPLIT" \
  "${EXTRA_ARGS[@]}" \
  --workers "$NUM_WORKERS" \
  --config "$DEST_CONFIG" \
  --output "$OUTPUT_DIR" 2>&1 \
  | grep -v "Cost calc failed for" \
  | tee "$OUTPUT_DIR/run.log"

EXIT_CODE=${PIPESTATUS[0]}
echo "[run_mini_eval] mini-extra swebench exit=$EXIT_CODE"

# 6. Stop vLLM and the diagnostic loop
kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true
kill "$DIAG_PID" 2>/dev/null || true

# 7. Drain tmpfs so condor's exit-time cgroup-memory poll doesn't flag a
#    "memory limit exceeded" hold despite our clean exit. /tmp is tmpfs and
#    counts toward the cgroup; rootless-docker overlay + vllm/triton caches
#    can leave 50-100GB resident even after mini-extra finishes.
echo "[run_mini_eval] draining tmpfs"
DOCKER_HOST=unix:///tmp/docker.sock docker stop $(DOCKER_HOST=unix:///tmp/docker.sock docker ps -q 2>/dev/null) 2>/dev/null || true
pkill -TERM -f dockerd 2>/dev/null || true
pkill -TERM -f containerd 2>/dev/null || true
sleep 2
rm -rf /tmp/docker /tmp/docker-data /tmp/dockerd-rootless /tmp/runc \
       /tmp/vllm_cache /tmp/triton /tmp/torchinductor \
       /tmp/modules /tmp/vllm.log /tmp/modelconfig_eval.yaml 2>/dev/null
sync
echo "[run_mini_eval] post-drain /tmp: $(du -sBM /tmp 2>/dev/null | cut -f1)"
sleep 3   # give condor procd one poll cycle to see lowered memory before exit

exit "$EXIT_CODE"
