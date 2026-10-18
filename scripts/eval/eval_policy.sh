#!/usr/bin/env bash
# Evaluate a multi-dance policy: every dance x (clean + each stress factor + gains,mass + all), then print a table.
# Run from ~/projects/unitree_rl_lab with env_isaaclab active:
#   bash ~/projects/g1_pilot/p2/eval_policy.sh <run_folder> [name] [task] [clips]
# defaults: name=p2, task=...-Residual-DR-D102-Wide-Multi2, clips="dance_102 gangnam"
# <run_folder> = folder under logs/rsl_rl/g1_schuhplattler_residual/ (its newest model_*.pt is used).
# Results: ~/projects/g1_pilot/d102/<name>__<clip>__<condition>/summary.txt ; table: ~/projects/g1_pilot/d102/<name>_table.txt
set -u
if [ $# -lt 1 ]; then echo "usage: $0 <run_folder> [name] [task] [clips]"; exit 1; fi
RUN=$1
NAME=${2:-p2}
TASK=${3:-Unitree-G1-29dof-Mimic-Schuhplattler-Residual-DR-D102-Wide-Multi2}
CLIPS=${4:-"dance_102 gangnam"}
OUT=$HOME/projects/g1_pilot/d102
CKPT=logs/rsl_rl/g1_schuhplattler_residual/$RUN
CONDS="none gains mass friction noise delay gains,mass all"
for clip in $CLIPS; do
  for c in $CONDS; do
    tag=${c//,/_}
    extra=""; [ "$c" != none ] && extra="--stress $c"
    echo "=== $NAME  $clip  $c"
    python ~/projects/g1_pilot/day4_eval.py --task $TASK --checkpoint $CKPT --clip $clip $extra \
      --out $OUT --run ${NAME}__${clip}__${tag} --headless 2>&1 | grep -E "^result|tracked_s|completed_pct" 
  done
done
python ~/projects/g1_pilot/p2/eval_table.py $OUT $NAME | tee $OUT/${NAME}_table.txt
