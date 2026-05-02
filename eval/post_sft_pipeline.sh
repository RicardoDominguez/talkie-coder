#!/bin/bash
# After an SFT run completes: repackage for vLLM, patch configs, register in
# litellm, submit 5x pass@5 condor jobs against the 42-instance subset.
#
# Usage:
#   post_sft_pipeline.sh <sft_output_dir> <vllm_dir_basename> <pass5_tag>
#
# Example:
#   post_sft_pipeline.sh \
#     /fast/rolmedo/swe-models/talkie-1930-it-coder-s20-e3 \
#     talkie-1930-it-coder-s20-e3-vllm \
#     talkie-1930-it-coder-s20-e3-pass5-union42
set -euo pipefail

SRC="${1:?sft_output_dir required}"
VLLM_BASENAME="${2:?vllm_dir_basename required}"
PASS5_TAG="${3:?pass5_tag required}"

DST="/fast/rolmedo/swe-models/${VLLM_BASENAME}"
INSTANCE_IDS=/home/rolmedo/talkie/sft/swe_bench_eval_union42.json

echo "==========================================================="
echo "[post_sft] SRC=$SRC"
echo "[post_sft] DST=$DST"
echo "[post_sft] PASS5_TAG=$PASS5_TAG"
echo "==========================================================="

# 1. Repackage (lm_head_gain bake, dd to /fast)
echo "[post_sft] repackage_for_vllm ..."
/home/rolmedo/tflatest/bin/python /lustre/home/rolmedo/talkie-eval/sft/repackage_for_vllm.py \
  --src "$SRC" --dst "$DST"

# 2. Copy refactored modeling, bust trust-remote-code cache
cp /lustre/home/rolmedo/talkie-eval/sft/modeling_talkie.py "$DST/modeling_talkie.py"
rm -rf /tmp/modules/transformers_modules/ 2>/dev/null || true

# 3. Patch config.json: add AutoModel to auto_map
python3 -c "
import json, sys
p = sys.argv[1]
c = json.load(open(p))
c.setdefault('auto_map', {})
c['auto_map']['AutoModel'] = 'modeling_talkie.TalkieModel'
json.dump(c, open(p,'w'), indent=2)
print('  patched auto_map.AutoModel ->', c['auto_map']['AutoModel'])
" "$DST/config.json"

# 4. Patch tokenizer_config.json: eos -> <|end|>
python3 -c "
import json, sys
p = sys.argv[1]
c = json.load(open(p))
c['eos_token'] = '<|end|>'
json.dump(c, open(p,'w'), indent=2)
print('  patched eos_token -> <|end|>')
" "$DST/tokenizer_config.json"

# 5. Generation config
cat > "$DST/generation_config.json" <<'EOF'
{
  "_from_model_config": true,
  "eos_token_id": [65536, 65535],
  "pad_token_id": 65535,
  "transformers_version": "4.57.3"
}
EOF
echo "  wrote generation_config.json"

# 6. Register in litellm price registry
LITELLM_KEY="hosted_vllm/${DST}"
python3 -c "
import json, sys
path = '/home/rolmedo/mini-swe-agent/model_prices_and_context_window.json'
key = sys.argv[1]
data = json.load(open(path))
if key not in data:
    data[key] = {
        'input_cost_per_token': 0,
        'output_cost_per_token': 0,
        'max_input_tokens': 32000,
        'litellm_provider': 'together_ai',
        'supports_function_calling': True,
        'supports_parallel_function_calling': True,
        'mode': 'chat',
        'supports_tool_choice': True,
        'source': 'talkie SFT minimal-sweep checkpoint',
    }
    json.dump(data, open(path,'w'), indent=4)
    print(f'  registered: {key}')
else:
    print(f'  already registered: {key}')
" "$LITELLM_KEY"

# 7. Submit pass@5 (5 condor sub-jobs, each runs all 42 instances)
echo "[post_sft] submitting pass@5 ..."
cd /lustre/home/rolmedo/talkie-eval/swe-evals
./launch_pass5_subset.sh "$DST" "$PASS5_TAG" "$INSTANCE_IDS" 5

echo "==========================================================="
echo "[post_sft] done. Eval dirs: /fast/rolmedo/swesmith/${PASS5_TAG}-run{1..5}/"
echo "==========================================================="
