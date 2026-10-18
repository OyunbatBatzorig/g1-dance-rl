#!/usr/bin/env bash
# "More motions, more robust?" at equal training: P1 / P2 / P3 at their ~5000-iteration checkpoints, Isaac Lab
# Monte Carlo inside the training DR and 1.5x wider, per dance. Same tests as robust_all.sh (mc, mc_w1.5), no sweeps.
# Run from ~/projects/unitree_rl_lab with env_isaaclab active:  bash ~/projects/g1_pilot/robust/iso5k.sh [num_envs=256]
# Results: ~/projects/g1_pilot/robust/iso5k_<policy>__<clip>__<test>/summary.txt ; table: robust/iso5k_table.txt
set -u
N=${1:-256}
L=logs/rsl_rl/g1_schuhplattler_residual
PY=~/projects/g1_pilot/robust/robust_eval.py
T=Unitree-G1-29dof-Mimic-Schuhplattler-Residual-DR-D102-Wide
run() {  # <name> <task> <checkpoint> <clip>
  for w in 1.0 1.5; do
    local test=mc; [ $w = 1.5 ] && test=mc_w1.5
    echo "=== $1  $4  $test"
    python $PY --task $2 --checkpoint $3 --clip $4 --mode mc --num_envs $N --widen $w --name ${1}__${4}__${test} --headless 2>&1 \
      | grep -E "^\[robust\] task|^success|^where|^termination|Error|error:"
  done
}
run iso5k_p1 $T        $L/2026-10-16_14-08-58_d102_p1_more/model_4998.pt dance_102
run iso5k_p2 $T-Multi2 $L/2026-10-16_15-40-56_d102_p2/model_5000.pt      dance_102
run iso5k_p2 $T-Multi2 $L/2026-10-16_15-40-56_d102_p2/model_5000.pt      gangnam
run iso5k_p3 $T-Multi3 $L/2026-10-16_19-34-00_d102_p3/model_5000.pt      dance_102
run iso5k_p3 $T-Multi3 $L/2026-10-16_19-34-00_d102_p3/model_5000.pt      gangnam
run iso5k_p3 $T-Multi3 $L/2026-10-16_19-34-00_d102_p3/model_5000.pt      schuhplattler
python ~/projects/g1_pilot/robust/robust_table.py iso5k_p1 iso5k_p2 iso5k_p3 | tee ~/projects/g1_pilot/robust/iso5k_table.txt
