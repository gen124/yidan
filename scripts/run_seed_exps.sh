#!/usr/bin/env bash
set -euo pipefail

# Sequentially run train1_extended for outputs/seed/exp1..exp3
# Each run uses nohup so it survives logout; logs to <out>/train.log.
# Adjust CONDA_ACTIVATE if needed for your system.

CONDA_ACTIVATE="source /home/hpc/anaconda3/bin/activate patch-3d"
BASE_DIR=$(pwd)

for i in 1 2 3; do
  OUT_DIR="$BASE_DIR/outputs/seed/exp${i}"
  CFG="$OUT_DIR/config.yaml"
  LOG="$OUT_DIR/train.log"

  mkdir -p "$OUT_DIR"

  echo "Starting exp${i}: config=$CFG, out=$OUT_DIR, log=$LOG"

  # activate conda env (best-effort)
  eval "$CONDA_ACTIVATE" || true

  # start with nohup in background and wait for completion before next
  nohup python train1_extended.py --config "$CFG" --out_dir "$OUT_DIR" --resume auto > "$LOG" 2>&1 &
  PID=$!
  echo "Launched PID $PID for exp${i}, waiting..."
  wait $PID
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "Run exp${i} exited with code $RC -- stopping sequence." >&2
    exit $RC
  fi
  echo "Finished exp${i} (PID $PID)"
done

echo "All runs completed."