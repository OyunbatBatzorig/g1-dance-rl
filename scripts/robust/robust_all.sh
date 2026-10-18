#!/usr/bin/env bash
# Robustness tests of one policy, per dance: Monte Carlo inside the training DR, Monte Carlo 1.5x wider,
# and one-factor sweeps (mass, gains, friction, delay) with everything else random. Then prints the table.
# Run from ~/projects/unitree_rl_lab with env_isaaclab active:
#   bash ~/projects/g1_pilot/robust/robust_all.sh <run_folder> <name> <task> "<clips>" [num_envs=256]
# <run_folder> = folder under logs/rsl_rl/g1_schuhplattler_residual/ (its newest model_*.pt is used)
# Results: ~/projects/g1_pilot/robust/<name>__<clip>__<test>/summary.txt ; table: robust/<name>_table.txt
set -u
if [ $# -lt 4 ]; then echo "usage: $0 <run_folder> <name> <task> \"<clips>\" [num_envs]"; exit 1; fi
RUN=$1; NAME=$2; TASK=$3; CLIPS=$4; N=${5:-256}
CKPT=logs/rsl_rl/g1_schuhplattler_residual/$RUN
PY=~/projects/g1_pilot/robust/robust_eval.py
run() {  # <clip> <test name> <extra args...>
  local clip=$1 test=$2; shift 2
  echo "=== $NAME  $clip  $test"
  python $PY --task $TASK --checkpoint $CKPT --clip $clip --name ${NAME}__${clip}__${test} --headless "$@" 2>&1 \
    | grep -E "^\[robust\] task|^success|^ +[0-9.]+ +[0-9.]+ +[0-9.]+|^  [a-z_]+ +\[|^where|^termination|Error|error:"
}
for clip in $CLIPS; do
  run $clip mc --mode mc --num_envs $N
  run $clip mc_w1.5 --mode mc --num_envs $N --widen 1.5
  for f in mass gains friction delay; do
    run $clip sweep_$f --mode sweep --factor $f --per_value 32
  done
done
python ~/projects/g1_pilot/robust/robust_table.py $NAME | tee ~/projects/g1_pilot/robust/${NAME}_table.txt
