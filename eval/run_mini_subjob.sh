#!/bin/bash
# Sub-job runner: receives MODEL_DIR, BASE_OUTPUT_DIR, JOB_ID, START, END as
# environment vars and runs mini-extra swebench on the given slice into a
# per-job output dir.
set -uo pipefail

# HTCondor strips the user's PATH; restore it.
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export USER="${USER:-$(id -un)}"
export HOME="${HOME:-/lustre/home/$USER}"
export PATH="$HOME/bin:$PATH"

# Required env vars
: "${MODEL_DIR:?MODEL_DIR must be set}"
: "${BASE_OUTPUT_DIR:?BASE_OUTPUT_DIR must be set}"
: "${JOB_ID:?JOB_ID must be set}"

# Two modes for instance selection:
#  - SLICE mode (legacy): START + END set -> mini-extra --slice $START:$END
#  - INSTANCE_IDS mode: pass a JSON file path; SLICE is left unset and the
#    underlying eval calls --instance-ids instead.
if [ -n "${INSTANCE_IDS:-}" ]; then
    export SLICE=""
    export INSTANCE_IDS
    echo "[run_mini_subjob] JOB_ID=$JOB_ID INSTANCE_IDS=$INSTANCE_IDS"
else
    : "${START:?START must be set when INSTANCE_IDS is not provided}"
    : "${END:?END must be set when INSTANCE_IDS is not provided}"
    export SLICE="${START}:${END}"
    echo "[run_mini_subjob] JOB_ID=$JOB_ID SLICE=$SLICE"
fi

export OUTPUT_DIR="$BASE_OUTPUT_DIR/job_$JOB_ID"
export NUM_WORKERS="${NUM_WORKERS:-16}"

mkdir -p "$OUTPUT_DIR"
echo "[run_mini_subjob] JOB_ID=$JOB_ID SLICE=$SLICE OUTPUT=$OUTPUT_DIR"

exec /lustre/home/rolmedo/talkie-eval/swe-evals/run_mini_eval.sh
