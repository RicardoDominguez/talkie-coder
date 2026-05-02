#!/bin/bash
# Wraps run_grade.py for HTCondor. Runs the swebench harness against a
# preds.json on the SWE-Bench-Verified-Working-Harbor dataset and writes a
# final report at OUTPUT_DIR/<RUN_ID>.json.
set -uo pipefail

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export USER="${USER:-$(id -un)}"
export HOME="${HOME:-/lustre/home/$USER}"
export PATH="$HOME/bin:$PATH"

: "${PREDS_FILE:?PREDS_FILE required}"
: "${OUTPUT_DIR:?OUTPUT_DIR required}"
: "${RUN_ID:?RUN_ID required}"
MAX_WORKERS="${MAX_WORKERS:-12}"

mkdir -p "$OUTPUT_DIR"
echo "[grade] PREDS=$PREDS_FILE OUTPUT=$OUTPUT_DIR RUN_ID=$RUN_ID WORKERS=$MAX_WORKERS"

# run_grade.py uses ./docker_setup.sh from cwd, so run from swe-evals dir
cd /lustre/home/rolmedo/talkie-eval/swe-evals

source /home/rolmedo/swa/bin/activate

python run_grade.py \
  --output_dir "$OUTPUT_DIR" \
  --preds_file "$PREDS_FILE" \
  --run_id "$RUN_ID" \
  --max_workers "$MAX_WORKERS" \
  --keep_logs

echo "[grade] done; report at $OUTPUT_DIR/$RUN_ID.json"
