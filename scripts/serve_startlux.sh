#!/usr/bin/env bash
# Serve StartLux-Decision-v1.1-27B-Instant as the local System-1 decision model:
# POST /v1/systemone on :8100 (the JevClient dialect: questions -> answers with
# choice + probabilities).
#
# Why this is not just `startlux-decide serve --launch-vllm`:
#   * the 27B weights are 51 GiB per GPU at TP=1 -- an 80GB A800 cannot also
#     hold the Instant models' WHOLE-prompt one-step prefill of a 128K context
#     (OOM at engine init), so the card must be split tensor-parallel;
#   * the package's launcher only does one-server-per-GPU (data parallel), no
#     TP flag -- so we start vLLM ourselves with the package's exact release
#     flags (see startlux_decide/launch.py server_command) plus
#     --tensor-parallel-size, and front it with `serve --vllm-url`.
# The prompt-bidirectional plugin (REQUIRED for Instant models -- plain vLLM
# silently serves a different model) is switched on via STARTLUX_PREFIX_BIDIR=1
# and the script aborts unless the server log shows it active.
#
# One-time setup (source lives in ~/projects, survives pod resets):
#   /opt/python/bin/pip install --no-deps -e /root/projects/StartLux-Decision
# Config: system1.backend: jev, system1.jev.base_url: http://localhost:8100
set -euo pipefail

MODEL="${MODEL:-/root/llm_models/StartLux-Decision-v1.1-27B-Instant}"
NAME="${NAME:-StartLux-Decision-v1.1-27B-Instant}"
PORT="${PORT:-8100}"            # decision server (/v1/systemone)
VLLM_PORT="${VLLM_PORT:-8001}"  # the vLLM server it fronts
GPUS="${GPUS:-0,1}"
TP="${TP:-2}"
MAX_LEN="${MAX_LEN:-131072}"    # the Instant release context; needs TP=2 on 80G cards
LOG="${LOG:-/root/llm_models/vllm_startlux_tp2.log}"

export VLLM_NO_USAGE_STATS=1
export STARTLUX_PREFIX_BIDIR=1

echo ">> vLLM: ${MODEL}  GPUs=${GPUS} TP=${TP} ctx=${MAX_LEN} :${VLLM_PORT}"
CUDA_VISIBLE_DEVICES="${GPUS}" /opt/python/bin/python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --served-model-name m \
  --host 127.0.0.1 --port "${VLLM_PORT}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_LEN}" \
  --gpu-memory-utilization 0.88 \
  --logprobs-mode processed_logprobs --max-logprobs 26 \
  --generation-config vllm \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --max-num-batched-tokens "${MAX_LEN}" \
  --limit-mm-per-prompt '{"image":8,"video":0}' \
  --mm-processor-kwargs '{"size":{"longest_edge":1048576,"shortest_edge":65536}}' \
  --trust-request-chat-template --skip-mm-profiling \
  --disable-uvicorn-access-log \
  > "${LOG}" 2>&1 &
VLLM_PID=$!

code=000
for _ in $(seq 1 120); do
  sleep 10
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "!! vLLM exited early; last lines of ${LOG}:"; tail -30 "${LOG}"; exit 1
  fi
  code=$(curl -s -m 3 -o /dev/null -w "%{http_code}" "http://127.0.0.1:${VLLM_PORT}/health" || true)
  [ "${code}" = "200" ] && break
done
if [ "${code}" != "200" ]; then
  echo "!! vLLM not healthy after 20 min; last lines of ${LOG}:"; tail -30 "${LOG}"; exit 1
fi
if ! grep -q "prefix-bidirectional prompt attention ON" "${LOG}"; then
  echo "!! prompt-bidirectional plugin NOT active in ${LOG}:"; grep -i plugin "${LOG}" | head -5; exit 1
fi
echo ">> vLLM healthy on :${VLLM_PORT}, plugin active"

echo ">> decision server :${PORT} fronting :${VLLM_PORT}"
exec /opt/python/bin/startlux-decide serve \
  --model "${MODEL}" \
  --name "${NAME}" \
  --vllm-url "http://127.0.0.1:${VLLM_PORT}" \
  --host 0.0.0.0 \
  --port "${PORT}"
