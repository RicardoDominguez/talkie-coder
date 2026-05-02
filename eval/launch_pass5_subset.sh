#!/bin/bash
# Pass@K eval against a fixed instance-id subset (small, e.g. 42 instances).
# Submits K independent condor sub-jobs, each runs ALL of the subset (no slicing).
# Each pass-run gets its own output dir for later merge + grade + union.
#
# Differs from launch_parallel_eval.sh:
#  - 1 sub-job per pass-run (vs 20 sub-jobs slicing 446 instances)
#  - --instance-ids replaces --slice; no need to fan out small subsets
#  - K-runs-of-the-same-thing for sampling-variance pass@K
#
# Usage:
#   launch_pass5_subset.sh <model_dir> <base_tag> <instance_ids_json> [k]
#
# Example:
#   launch_pass5_subset.sh \
#     /fast/rolmedo/swe-models/talkie-1930-it-coder-s20-e3-vllm \
#     talkie-1930-it-coder-s20-e3-pass5-union42 \
#     /home/rolmedo/talkie/sft/swe_bench_eval_union42.json \
#     5
set -euo pipefail

MODEL_DIR="${1:?model_dir required}"
BASE_TAG="${2:?base_tag (used as /fast/rolmedo/swesmith/<tag>-runI dir prefix)}"
INSTANCE_IDS="${3:?instance_ids JSON path required}"
K="${4:-5}"

[ -f "$INSTANCE_IDS" ] || { echo "instance_ids file not found: $INSTANCE_IDS"; exit 1; }

CONDOR_LOG_DIR=/lustre/home/rolmedo/talkie-eval/swe-evals/condor_logs
mkdir -p "$CONDOR_LOG_DIR"

echo "[launch_pass5_subset] MODEL=$MODEL_DIR"
echo "[launch_pass5_subset] BASE_TAG=$BASE_TAG"
echo "[launch_pass5_subset] INSTANCE_IDS=$INSTANCE_IDS (K=$K parallel pass-runs)"

for (( i=1; i<=K; i++ )); do
    BASE_OUTPUT_DIR="/fast/rolmedo/swesmith/${BASE_TAG}-run${i}"
    mkdir -p "$BASE_OUTPUT_DIR"
    SUB_FILE="$BASE_OUTPUT_DIR/job_0.sub"
    cat > "$SUB_FILE" <<EOF
executable = /lustre/home/rolmedo/talkie-eval/swe-evals/run_mini_subjob.sh

environment = "MODEL_DIR=$MODEL_DIR BASE_OUTPUT_DIR=$BASE_OUTPUT_DIR JOB_ID=0 INSTANCE_IDS=$INSTANCE_IDS NUM_WORKERS=5"

request_cpus   = 8
request_gpus   = 1
request_memory = 256GB
request_disk   = 300GB

requirements = (TARGET.CUDACapability == 9.0 || TARGET.CUDACapability == 10.0) && (TARGET.UtsnameNodename != "i206") && (TARGET.UtsnameNodename != "g105") && (TARGET.UtsnameNodename != "g174") && (TARGET.CUDADriverVersion >= 13.0)
+BypassLXCfs = true

# Self-remove if running > 2h (stuck vLLM / mini-extra trajectory).
periodic_remove = (JobStatus == 2) && ((CurrentTime - EnteredCurrentStatus) > 7200)
periodic_remove_reason = "Wallclock exceeded 2h cap"

output = $CONDOR_LOG_DIR/sub.\$(Cluster).\$(Process).out
error  = $CONDOR_LOG_DIR/sub.\$(Cluster).\$(Process).err
log    = $CONDOR_LOG_DIR/sub.\$(Cluster).log

queue
EOF
    echo "  submitting pass-run $i -> $BASE_OUTPUT_DIR"
    condor_submit_bid 51 "$SUB_FILE" 2>&1 | tail -1
    sleep 1   # stagger
done

echo "[launch_pass5_subset] all $K pass-runs submitted"
