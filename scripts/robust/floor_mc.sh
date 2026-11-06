#!/bin/bash
# Isaac MC per floor type (training DR, friction 0.3-2.0): floor (model_34998) vs plus (model_29999), 512 robots.
# Each twice: with the training terminations, and without ee_body_pos (real falls only).
cd ~/projects/unitree_rl_lab
L=~/projects/unitree_rl_lab/logs/rsl_rl/g1_schuhplattler_residual
T=Unitree-G1-29dof-Mimic-Schuhplattler-MimicAction-DRFloor-SchV3s
declare -A CK=([floor]=$L/2026-10-31_22-40-36_sch_v3s_floor/model_34998.pt [plus]=$L/2026-10-30_20-26-10_sch_v3s_plus/model_29999.pt)
for fl in flat rough stone slope; do
  for pol in floor plus; do
    for ee in on off; do
      x=""; [ $ee = off ] && x="--no_ee_term"
      n=v3s_${pol}__${fl}__ee${ee}
      echo "== $n"
      conda run --no-capture-output -n env_isaaclab python ~/projects/g1_pilot/robust/robust_eval.py --task $T \
        --checkpoint ${CK[$pol]} --mode mc --num_envs 512 --floor $fl $x --name $n --headless 2>&1 \
        | grep -E "^success|^where|^termination|Traceback|Error:" 
    done
  done
done
echo "== ALL_DONE"
