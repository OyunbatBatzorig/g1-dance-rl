#!/bin/bash
# Set how far Mimic_Schuhplattler plays before it hands over to Velocity (seconds of the intro CSV).
# On the robot: 16.96 only (the full dance ends in the Velocity stance). Shorter values only in the simulator:
# at an early stop the Velocity policy snaps the arms back (3.66: arms 2.7 rad overhead).
# The simulator and the real robot use the same config.yaml: set it back to a safe point before run_real.sh.
#   bash ~/projects/g1_pilot/sim_cpp/set_time_end.sh 3.66
CFG="$(cd "$(dirname "$0")" && pwd)/deploy/robots/g1_29dof/config/config.yaml"
[ -z "$1" ] && { echo "usage: set_time_end.sh <seconds>   robot: 16.96 (full dance)"; exit 1; }
[ "$1" = "16.96" ] || echo "WARNING: $1 stops the dance early (Velocity snaps the arms back): simulator only, run_real.sh refuses it"
python3 - "$CFG" "$1" <<'PY'
import sys, re
p, te = sys.argv[1], sys.argv[2]
s = open(p).read()
key = "\n  Mimic_Schuhplattler:\n    transitions:"
i = s.index(key); j = s.index("\n  Mimic_Schuhplattler_ArmTest:\n    transitions:", i)
blk = re.sub(r"time_end: [\d.]+", "time_end: " + te, s[i:j])
open(p, "w").write(s[:i] + blk + s[j:])
print(f"Mimic_Schuhplattler time_end = {te} s  ({p})")
PY
