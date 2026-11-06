#!/bin/bash
# deploy-equivalent MuJoCo checks of deploy/schuhplattler_v3s, 512 runs each (training DR, seed 0)
cd ~/projects/g1_pilot
R="conda run --no-capture-output -n unitree_sim_env python deploy/deploy_sim2sim.py --pkg deploy/schuhplattler_v3s --runs 512 --procs 22 --no_fall_stop"
run() { echo "== $1"; $R --name "$@" 2>&1 | grep -E "nominal|Monte|terminations|Traceback|Error"; }
case "$1" in
  ref)
    run dep_ref_x1                                  # = sim2sim_dance v3s_floor_flat_n512, but ONNX + deploy obs
    run dep_ref_x1_w15 --widen 1.5 ;;
  *)
    run dep_stand_x1 --start stand                  # gamepad start from the default stance, no intro
    run dep_stand_intro2_x1 --start stand --intro 2
    for f in 45 37.5 25; do run dep_stand_intro2_fps$f --start stand --intro 2 --fps $f; done ;;
esac
echo "== ALL_DONE"
