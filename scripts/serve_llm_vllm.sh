#!/usr/bin/env bash
# Serve the System-2 LLM (Qwen3.8-27B) with vLLM's OpenAI-compatible API on :8000.
#
# GPU layout (8x A800): 6 GPUs for this S2 server, 2 for the S1 decision server
# (serve_startlux.sh, GPUs 0,1). Qwen3.8-27B has 4 KV heads, so tensor parallel
# must divide 4 -- TP=6 is NOT valid. The 6-GPU budget is spent as DP=3 replicas
# x TP=2: vLLM's internal data-parallel load balancer multiplexes :8000 across
# the replicas (~3x the decode throughput of a single TP=2 server; the S2 review
# queue needs it with thinking enabled).
# Config: system2.backend: llm, system2.llm.base_url: http://localhost:8000/v1,
#         system2.llm.model: Qwen3.8-27B
set -euo pipefail

MODEL="${MODEL:-/root/llm_models/Qwen3.8-27B}"
SERVED_NAME="${SERVED_NAME:-Qwen3.8-27B}"
PORT="${PORT:-8000}"
GPU_IDS="${GPU_IDS:-2,3,4,5,6,7}"
TP="${TP:-2}"
DP="${DP:-3}"
MAX_LEN="${MAX_LEN:-131072}"   # 128K context

echo ">> Serving S2 LLM '${MODEL}' as '${SERVED_NAME}' on :${PORT} (DP=${DP} x TP=${TP}, GPUs=${GPU_IDS})"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  "${VLLM:-/opt/python/bin/vllm}" serve "${MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --data-parallel-size "${DP}" \
    --max-model-len "${MAX_LEN}" \
    --reasoning-parser qwen3 \
    --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "I have to give the solution based on the reasoning directly now.</think>"}' \
    --trust-remote-code
