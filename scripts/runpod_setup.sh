#!/usr/bin/env bash
# Provision a fresh RunPod GPU pod for Omnispan's vLLM path.
#
# Run ON THE POD (see scripts/RUNPOD.md for how to connect):
#   curl -sSL https://raw.githubusercontent.com/db42/omnispan/main/scripts/runpod_setup.sh | bash
# or, if the repo is already cloned:
#   bash /root/omnispan/scripts/runpod_setup.sh
#
# Idempotent: safe to re-run. Everything lands on the container disk (50 GB),
# not /workspace (20 GB) -- model weights do not fit on the volume.
#
# Env:
#   MODEL_ID       model to pre-download (default Qwen/Qwen3-32B-AWQ; "" to skip)
#   SKIP_ENGINE=1  skip the Rust toolchain + engine build
set -euo pipefail

REPO_DIR=${REPO_DIR:-/root/omnispan}
REPO_URL=${REPO_URL:-https://github.com/db42/omnispan.git}
MODEL_ID=${MODEL_ID-Qwen/Qwen3-32B-AWQ}

log() { echo -e "\n=== $* ==="; }

log "GPU / disk"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "no nvidia-smi"
df -h / | tail -1

log "Repo -> $REPO_DIR"
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$REPO_DIR"
fi
git -C "$REPO_DIR" log --oneline -1

log "Python deps (vLLM)"
if python -c "import vllm" 2>/dev/null; then
  echo "vllm already installed: $(python -c 'import vllm;print(vllm.__version__)')"
else
  pip install --no-cache-dir vllm grpcio grpcio-tools
fi
python -c "import vllm; print('vllm', vllm.__version__)"

if [ "${SKIP_ENGINE:-0}" != "1" ]; then
  log "Rust toolchain + engine build"
  if ! command -v cargo >/dev/null 2>&1; then
    curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
  fi
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
  cargo --version
  (cd "$REPO_DIR/engine" && cargo build --release)
  ls -la "$REPO_DIR/engine/target/release/omnispan-engine"
fi

if [ -n "$MODEL_ID" ]; then
  log "Pre-downloading weights: $MODEL_ID"
  # vLLM would fetch these implicitly on first load, which looks like a hang.
  # Doing it explicitly makes progress visible and failures obvious.
  pip install -q "huggingface_hub[cli]"
  hf download "$MODEL_ID"
  df -h / | tail -1
fi

log "API compatibility check"
python "$REPO_DIR/scripts/probe_vllm_api.py" || echo "probe reported issues (see above)"

log "Done"
cat <<EOF
Next steps (two terminals on the pod):

  # 1. worker
  cd $REPO_DIR
  WORKER_BACKEND=vllm VLLM_ASYNC=1 MODEL_ID=${MODEL_ID:-<model>} \\
    VLLM_QUANTIZATION=AWQ VLLM_MAX_MODEL_LEN=4096 VLLM_GPU_MEMORY_UTILIZATION=0.85 \\
    python worker/worker.py

  # 2. engine
  cd $REPO_DIR/engine
  ENGINE_MODE=queued WORKER_ENDPOINT=http://127.0.0.1:50071 ./target/release/omnispan-engine

  # 3. benchmark (on the pod)
  cd $REPO_DIR
  python bench/benchmark.py --requests 8 --concurrency 8 --max-tokens 150 --stream --mode vllm_stream
EOF
