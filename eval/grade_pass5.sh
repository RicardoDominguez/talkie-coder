#!/bin/bash
# Merge preds + submit grade jobs for a pass@5 set. Idempotent: skips
# pass-runs whose preds.merged.json or final report already exist.
#
# Usage:
#   grade_pass5.sh <pass5_tag> [k]
#
# Example:
#   grade_pass5.sh talkie-1930-it-coder-s20-e3-pass5-union42 5
set -euo pipefail

TAG="${1:?pass5_tag required}"
K="${2:-5}"

CONDOR_LOG_DIR=/lustre/home/rolmedo/talkie-eval/swe-evals/condor_logs
mkdir -p "$CONDOR_LOG_DIR"

for (( i=1; i<=K; i++ )); do
    BASE="/fast/rolmedo/swesmith/${TAG}-run${i}"
    if [ ! -d "$BASE" ]; then
        echo "[grade_pass5] run${i}: $BASE missing, skipping"
        continue
    fi
    RUN_ID="${TAG}-run${i}"
    REPORT="$BASE/${RUN_ID}.json"
    if [ -f "$REPORT" ]; then
        echo "[grade_pass5] run${i}: report already exists ($REPORT), skipping"
        continue
    fi

    # 1. Merge preds.json across sub-job dirs (latest-wins, drop non-diff)
    echo "[grade_pass5] run${i}: merging preds ..."
    python3 -c "
import json, os, glob
paths = sorted(glob.glob('$BASE/*/preds.json'), key=os.path.getmtime, reverse=True)
merged = {}
for p in paths:
    try: d = json.load(open(p))
    except: continue
    for inst, rec in d.items():
        if inst in merged: continue
        if rec.get('model_patch','').startswith('diff --git'):
            merged[inst] = rec
print(f'  merged {len(merged)} valid predictions from {len(paths)} sub-job preds.json files')
json.dump(merged, open('$BASE/preds.merged.json','w'), indent=2)
"

    # 2. Write per-run grade .sub
    SUB_FILE="$BASE/grade.sub"
    cat > "$SUB_FILE" <<EOF
executable = /lustre/home/rolmedo/talkie-eval/swe-evals/run_grade_wrapper.sh

environment = "PREDS_FILE=$BASE/preds.merged.json OUTPUT_DIR=$BASE RUN_ID=$RUN_ID MAX_WORKERS=12"

request_cpus   = 16
request_gpus   = 0
request_memory = 200GB
request_disk   = 300GB

requirements = (TARGET.UtsnameNodename != "i206") && (TARGET.UtsnameNodename != "g105") && (TARGET.UtsnameNodename != "g174")
+BypassLXCfs = true

output = $CONDOR_LOG_DIR/grade.\$(Cluster).\$(Process).out
error  = $CONDOR_LOG_DIR/grade.\$(Cluster).\$(Process).err
log    = $CONDOR_LOG_DIR/grade.\$(Cluster).log

queue
EOF
    echo "[grade_pass5] run${i}: submitting grade job"
    condor_submit_bid 51 "$SUB_FILE" 2>&1 | tail -1
    sleep 1
done

echo "[grade_pass5] done; final reports will land at /fast/rolmedo/swesmith/${TAG}-run{1..${K}}/${TAG}-run<i>.json"
