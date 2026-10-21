#!/usr/bin/env bash
# Crash-tolerant training loop: resumes from the latest checkpoint if the
# process dies (e.g. OOM on a shared machine) until the step budget is used.
set -u
CONFIG=${1:-experiments/configs/ppo_cnn_v1.json}
DEADLINE=$(( $(date +%s) + ${2:-1100} ))
export PYTHONPATH=.
PY=${PYTHON:-python}
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  LEFT=$(( DEADLINE - $(date +%s) ))
  timeout -s TERM "$LEFT" "$PY" -u -m ai.train --config "$CONFIG" --resume auto ${INIT:+--init "$INIT"} && break
  sleep 2
done
