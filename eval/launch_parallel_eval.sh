#!/bin/bash
# Submit N parallel condor sub-jobs that together cover slice 0:TOTAL of
# SWE-bench Verified for a given MODEL_DIR.
#
# Usage:
#   launch_parallel_eval.sh <model_dir> <base_output_dir> [n_jobs] [total]
#
# Example:
#   launch_parallel_eval.sh \
#     /fast/rolmedo/swe-models/talkie-web-12h-v2-ckpt400-vllm \
#     /fast/rolmedo/swesmith/talkie-web-v2-ckpt400-mini-100 \
#     5 100
set -euo pipefail

MODEL_DIR="${1:?model_dir required}"
BASE_OUTPUT_DIR="${2:?base_output_dir required}"
N_JOBS="${3:-5}"
TOTAL="${4:-100}"
PER_JOB=$(( (TOTAL + N_JOBS - 1) / N_JOBS ))

mkdir -p "$BASE_OUTPUT_DIR"

CONDOR_LOG_DIR=/lustre/home/rolmedo/talkie-eval/swe-evals/condor_logs
mkdir -p "$CONDOR_LOG_DIR"

echo "[launch_parallel_eval] MODEL=$MODEL_DIR"
echo "[launch_parallel_eval] OUTPUT=$BASE_OUTPUT_DIR"
echo "[launch_parallel_eval] N_JOBS=$N_JOBS TOTAL=$TOTAL PER_JOB=$PER_JOB"

for (( i=0; i<N_JOBS; i++ )); do
    START=$((i * PER_JOB))
    END=$(( (i + 1) * PER_JOB ))
    if [ $END -gt $TOTAL ]; then END=$TOTAL; fi
    if [ $START -ge $END ]; then continue; fi

    SUB_FILE="$BASE_OUTPUT_DIR/job_$i.sub"
    cat > "$SUB_FILE" <<EOF
executable = /lustre/home/rolmedo/talkie-eval/swe-evals/run_mini_subjob.sh

environment = "MODEL_DIR=$MODEL_DIR BASE_OUTPUT_DIR=$BASE_OUTPUT_DIR JOB_ID=$i START=$START END=$END NUM_WORKERS=5"

request_cpus   = 8
request_gpus   = 1
request_memory = 256GB
request_disk   = 300GB

requirements = (TARGET.CUDACapability == 9.0 || TARGET.CUDACapability == 10.0) && (TARGET.UtsnameNodename != "i206") && (TARGET.UtsnameNodename != "g105") && (TARGET.UtsnameNodename != "g174")
+BypassLXCfs = true

# Self-remove if running > 2h (stuck vLLM / mini-extra trajectory). Job goes
# to Removed; trajectories already on /fast survive (mini-extra is idempotent).
periodic_remove = (JobStatus == 2) && ((CurrentTime - EnteredCurrentStatus) > 7200)
periodic_remove_reason = "Wallclock exceeded 2h cap"

output = $CONDOR_LOG_DIR/sub.\$(Cluster).\$(Process).out
error  = $CONDOR_LOG_DIR/sub.\$(Cluster).\$(Process).err
log    = $CONDOR_LOG_DIR/sub.\$(Cluster).log

queue
EOF
    echo "  submitting job_$i (slice $START:$END)"
    condor_submit_bid 51 "$SUB_FILE" 2>&1 | tail -1
    # 1s stagger so the cluster doesn't try to start all sub-jobs at once
    # — reduces simultaneous reads from /fast (vLLM weights + docker tarballs).
    sleep 1
done

echo "[launch_parallel_eval] all $N_JOBS sub-jobs submitted"
