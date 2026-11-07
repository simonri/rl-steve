#!/bin/bash
# The network experiments (docs/training-design.md#natet): from one model, the same number of decisions with each
# option and without any, one after another; then the fixed benchmark of each, all into runs/models/experiments.jsonl.
#
#   tools/run_experiments.sh START.zip LEVELS.json [decisions]
set -u
cd "$(dirname "$0")/.."
START=$(realpath "$1")
LEVELS=$(realpath "$2")
STEPS=${3:-8000000}
OUT=../runs/models
PY="uv run --no-project --python 3.12 --with torch --with stable-baselines3 --with sb3-contrib --with gymnasium python -u"
echo "start $(date)" >> $OUT/experiments.log
$PY benchmark.py "$START" --out $OUT/experiments.jsonl >> $OUT/experiments.log 2>&1
for exp in "base:" "critic:--critic-memory" "norm:--normalize-inputs" "entities:--entities"; do
  name="exp_${exp%%:*}"
  flags="${exp#*:}"
  rm -rf "$OUT/$name"
  echo "$name $flags $(date)" >> $OUT/experiments.log
  $PY train_sim.py --name "$name" --resume "$START" --levels "$LEVELS" --steps "$STEPS" $flags > "$OUT/$name.log" 2>&1
  $PY benchmark.py "$OUT/$name/latest.zip" --out $OUT/experiments.jsonl >> $OUT/experiments.log 2>&1
done
echo "done $(date)" >> $OUT/experiments.log
