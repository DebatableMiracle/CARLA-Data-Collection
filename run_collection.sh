#!/bin/bash
# run_collection.sh
# ─────────────────
# Launch CARLA server then start data collection.
# Run from the carla_vla/ project directory.

set -e

CARLA_ROOT="${CARLA_ROOT:-$HOME/carla}"
VENV="${VENV:-$HOME/venvs/carla_vla}"
GPU_ID=0   # Pin to GPU 0, leave GPU 1 idle

echo "==================================================="
echo "  CARLA VLA Data Collection Pipeline"
echo "  CARLA: $CARLA_ROOT"
echo "  GPU:   $GPU_ID"
echo "==================================================="

# ── 1. Start CARLA server (background) ──────────────────────────────────────
echo "[1/3] Starting CARLA server..."
CUDA_VISIBLE_DEVICES=$GPU_ID "$CARLA_ROOT/CarlaUE4.sh" \
    -RenderOffScreen \
    -quality-level=Epic \
    -carla-server \
    -benchmark \
    -fps=20 \
    2>&1 | tee /tmp/carla_server.log &

CARLA_PID=$!
echo "  CARLA PID: $CARLA_PID"

# ── 2. Wait for CARLA to be ready ────────────────────────────────────────────
echo "[2/3] Waiting for CARLA to boot..."
source "$VENV/bin/activate"
for i in $(seq 1 30); do
    sleep 2
    if python3 -c "
import carla, sys
try:
    c = carla.Client('127.0.0.1', 2000)
    c.set_timeout(5)
    v = c.get_server_version()
    print(f'  CARLA ready: v{v}')
    sys.exit(0)
except:
    sys.exit(1)
" 2>/dev/null; then
        break
    fi
    echo "  ... still waiting ($i/30)"
done

# ── 3. Run collection ─────────────────────────────────────────────────────────
echo "[3/3] Starting data collection..."
CUDA_VISIBLE_DEVICES=$GPU_ID python3 collect_data.py \
    --host 127.0.0.1 \
    --port 2000 \
    --tm-port 8000 \
    --episodes 500 \
    --steps-per-episode 600 \
    --fps 20 \
    --warmup-steps 30 \
    --output-dir ./data/episodes \
    --maps Town01 Town02 Town03 Town05 Town10HD \
    --map-switch-every 20 \
    --n-vehicles 40 \
    --n-walkers 20 \
    --perturb-prob 0.05 \
    --perturb-std 0.08 \
    --seed 42

echo "Collection complete."

# Kill CARLA server
kill $CARLA_PID 2>/dev/null || true
