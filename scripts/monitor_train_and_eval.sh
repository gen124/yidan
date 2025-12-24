
#!/usr/bin/env bash
# Monitor focal training process and run eval when it finishes
# Behavior: cd to repo root, activate conda env, poll for train process, run eval when train stops.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_PATTERN='train.py --config outputs/exp4-focal/config.yaml'
LOG="$REPO_ROOT/outputs/exp4-focal/monitor.log"

mkdir -p "$(dirname "$LOG")"
echo "[$(date -Iseconds)] Monitor started. Repo: $REPO_ROOT" >> "$LOG"

cd "$REPO_ROOT" || exit 1

while true; do
  # Use pgrep to match full cmdline; fallback to ps+grep if pgrep missing
  if command -v pgrep >/dev/null 2>&1; then
    if pgrep -f "$TRAIN_PATTERN" >/dev/null 2>&1; then
      echo "[$(date -Iseconds)] train running..." >> "$LOG"
      sleep 60
      continue
    fi
  else
    if ps aux | grep "$TRAIN_PATTERN" | grep -v grep >/dev/null 2>&1; then
      echo "[$(date -Iseconds)] train running..." >> "$LOG"
      sleep 60
      continue
    fi
  fi

  echo "[$(date -Iseconds)] train not found. Triggering evaluation..." >> "$LOG"
  # Activate conda env and run eval with unbuffered output
  # Use bash -lc to ensure activation runs in same shell context
  bash -lc "source /home/sunyidan/miniconda3/bin/activate /home/sunyidan/miniconda3/envs/patchmil-gpu && cd '$REPO_ROOT' && python -u scripts/eval_and_compare_all.py" >> "$LOG" 2>&1
  echo "[$(date -Iseconds)] Evaluation finished." >> "$LOG"
  exit 0
done
