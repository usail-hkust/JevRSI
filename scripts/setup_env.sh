#!/usr/bin/env bash
# Create the conda env and install everything for ALFWorld +
# MiniGrid (BabyAI-Text).
set -euo pipefail

ENV_NAME="${ENV_NAME:-jevmaster}"
PY_VERSION="${PY_VERSION:-3.10}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo ">> Creating conda env '${ENV_NAME}' (python ${PY_VERSION})"
conda create -n "${ENV_NAME}" "python=${PY_VERSION}" -y

conda install -n "${ENV_NAME}" -c conda-forge openjdk=11 -y

echo ">> Installing the jevmaster package + environments"
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install -e "${REPO_DIR}[alfworld,minigrid]"

echo ">> Installing the BabyAI-Text fork of Minigrid (BALROG protocol)"
# The PyPI minigrid package lacks the text-description observations the
# minigrid adapter uses; clone the BALROG fork next to this repo and install
# it editable (pulls gymnasium + pygame).
MINIGRID_DIR="${MINIGRID_DIR:-$(dirname "${REPO_DIR}")/Minigrid}"
if [ ! -d "${MINIGRID_DIR}" ]; then
  git clone https://github.com/BartekCupial/Minigrid "${MINIGRID_DIR}"
fi
conda run -n "${ENV_NAME}" python -m pip install -e "${MINIGRID_DIR}"

echo ">> Downloading ALFWorld data (set ALFWORLD_DATA to control location)"
export ALFWORLD_DATA="${ALFWORLD_DATA:-${REPO_DIR}/alfworld_data}"
echo "   ALFWORLD_DATA=${ALFWORLD_DATA}"
conda run -n "${ENV_NAME}" alfworld-download || \
  echo "   (alfworld-download failed or skipped)"

echo ">> Done. Activate with: conda activate ${ENV_NAME}"
echo "   Remember to export ALFWORLD_DATA=${ALFWORLD_DATA} for ALFWorld runs."
