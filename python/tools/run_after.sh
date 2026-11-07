#!/bin/bash
# After the network experiments: the HL-Gauss value from the same model, then an exploiter against that model.
#   tools/run_after.sh START.zip LEVELS.json [decisions]
set -u
cd "$(dirname "$0")/.."
START=$(realpath "$1")
LEVELS=$(realpath "$2")
STEPS=${3:-3000000}
OUT=../runs/models
PY="uv run --no-project --python 3.12 --with torch --with stable-baselines3 --with sb3-contrib --with gymnasium python -u"
while pgrep -f "tools/run_experiments.sh" > /dev/null; do sleep 30; done
rm -rf "$OUT/exp_hlgauss" "$OUT/exploit_v4"
echo "exp_hlgauss --hl-gauss $(date)" >> $OUT/experiments.log
$PY train_sim.py --name exp_hlgauss --resume "$START" --levels "$LEVELS" --steps "$STEPS" --no-evals --hl-gauss > "$OUT/exp_hlgauss.log" 2>&1
$PY benchmark.py "$OUT/exp_hlgauss/latest.zip" --modes sampled --out $OUT/experiments.jsonl >> $OUT/experiments.log 2>&1
echo "exploit_v4 $(date)" >> $OUT/experiments.log
$PY train_sim.py --name exploit_v4 --resume "$START" --levels "$LEVELS" --steps "$STEPS" --no-evals --exploit "$START" > "$OUT/exploit_v4.log" 2>&1
grep "exploiter against" "$OUT/exploit_v4.log" >> $OUT/experiments.log
echo "after done $(date)" >> $OUT/experiments.log
