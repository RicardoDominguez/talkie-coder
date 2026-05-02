#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <hostname> <port> <model_dir>"
  exit 1
fi

HOSTNAME="$1"
PORT="$2"
MODEL_DIR="$3"

# SRC="/home/rolmedo/mini-swe-agent/localconfig_no_model-qwen3-smith.yaml"

if [[ -z "${DEST:-}" ]]; then
  DEST="/tmp/modelconfig.yaml"
fi

if [[ -z "${TEMPERATURE:-}" ]]; then
  TEMPERATURE="0.7"
fi

cp "$SRC" "$DEST"

cat >> "$DEST" <<EOF

model:
  model_name: "hosted_vllm/${MODEL_DIR}"
  model_class: "qwen3"
  litellm_model_registry: "/home/rolmedo/mini-swe-agent/model_prices_and_context_window.json"
  model_kwargs:
    api_base: "http://${HOSTNAME}:${PORT}/v1"
    temperature: ${TEMPERATURE}
EOF

if [[ -n "${MAX_TOKENS:-}" ]]; then
  cat >> "$DEST" <<EOF
    max_tokens: ${MAX_TOKENS}
EOF
fi

if [[ -n "${REPETITION_PENALTY:-}" ]]; then
  cat >> "$DEST" <<EOF
    extra_body:
      repetition_penalty: ${REPETITION_PENALTY}
EOF
fi
