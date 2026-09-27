#!/usr/bin/env bash
# Train the queued leaves one at a time. Sequential on purpose: two encoders on 8 CPUs
# measured worse than one, and a failed leaf must not stop the rest.
cd /home/roni/Roni_workspace/layajev-mcp/oculus_swarm || exit 1
PY=/home/roni/Roni_workspace/layajev-mcp/.venv-swarm/bin/python
while read -r LEAF; do
  [ -z "$LEAF" ] && continue
  echo "=== $(date -Is) START $LEAF ==="
  timeout 3600 "$PY" -u train_on_dataset.py --leaf "$LEAF" --epochs 4 --holdout 0.25 \
    2>&1 | grep -vE "Fetching|RuntimeWarning|return Agent|UserWarning|warnings.warn|Triggered|it/s"
  echo "=== $(date -Is) END $LEAF rc=$? ==="
done < train_queue.txt
echo "=== $(date -Is) QUEUE DONE ==="
