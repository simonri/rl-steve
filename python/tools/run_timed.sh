#!/bin/bash
# Settings compared at the same wall-clock time (docs/training-design.md#fem): each run continues START for MINUTES,
# then the sampled benchmark, into ../runs/models/experiments.jsonl and experiments.log. Waits for running experiments.
#   tools/run_timed.sh START.zip LEVELS.json MINUTES "name args..." ["name args..."]...
set -u
cd "$(dirname "$0")/.."
START=$(realpath "$1"); LEVELS=$(realpath "$2"); MINUTES=$3; shift 3
OUT=../runs/models
PY="uv run --no-project --python 3.12 --with torch --with stable-baselines3 --with sb3-contrib --with gymnasium python -u"
while pgrep -f "train_sim.py|benchmark.py" > /dev/null; do sleep 30; done
for run in "$@"; do
  name=${run%% *}; args=${run#"$name"}
  rm -rf "$OUT/$name"
  echo "$name$args ${MINUTES}min $(date)" >> $OUT/experiments.log
  $PY train_sim.py --name "$name" --resume "$START" --levels "$LEVELS" --steps 1000000000 --minutes "$MINUTES" --no-evals $args > "$OUT/$name.log" 2>&1
  grep -E "minutes: stopped" "$OUT/$name.log" >> $OUT/experiments.log
  $PY benchmark.py "$OUT/$name/latest.zip" --modes sampled --out $OUT/experiments.jsonl >> $OUT/experiments.log 2>&1
done
echo "timed done $(date)" >> $OUT/experiments.log
