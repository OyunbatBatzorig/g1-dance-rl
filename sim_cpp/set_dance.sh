#!/bin/bash
# Choose what Mimic_Schuhplattler plays (g1_ctrl reads it at start: restart run_sim.sh / run_real.sh after this).
#   bash ~/projects/g1_pilot/sim_cpp/set_dance.sh full     # 2 s intro + whole dance, ends in the stance (16.96 s)
#   bash ~/projects/g1_pilot/sim_cpp/set_dance.sh part     # 2 s intro + first 1.66 s + 2 s calm return + 1 s (6.66 s)
CFG="$(cd "$(dirname "$0")" && pwd)/deploy/robots/g1_29dof/config/config.yaml"
case "$1" in
  full) CSV=schuhplattler_gmr_v3s_50hz_intro2.csv; TE=16.96 ;;
  part) CSV=schuhplattler_v3s_part01.66_50hz.csv; TE=6.66 ;;
  *) echo "usage: set_dance.sh full|part"; exit 1 ;;
esac
python3 - "$CFG" "$CSV" "$TE" <<'PY'
import sys, re
p, csv, te = sys.argv[1:4]
s = open(p).read()
i = s.index("\n  Mimic_Schuhplattler:\n    transitions:"); j = s.index("\n  Mimic_Schuhplattler_ArmTest:\n    transitions:", i)
blk = re.sub(r"time_end: [\d.]+", "time_end: " + te, s[i:j])
blk = re.sub(r"(motion_file: config/policy/mimic/schuhplattler_v3s/params/)\S+", r"\g<1>" + csv, blk)
open(p, "w").write(s[:i] + blk + s[j:])
print(f"Mimic_Schuhplattler plays {csv} until {te} s")
PY
