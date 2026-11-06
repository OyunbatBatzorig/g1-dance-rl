#!/bin/bash
# C++ sim2sim of the real deploy stack: unitree_mujoco (keyboard gamepad copy) + g1_ctrl (deploy copy with the
# Schuhplattler states). Both are copies under ~/projects/g1_pilot/sim_cpp; unitree_mujoco and unitree_rl_lab are untouched.
#
#   bash ~/projects/g1_pilot/sim_cpp/run_sim.sh              # interactive: click the MuJoCo window, then the keys below
#   bash ~/projects/g1_pilot/sim_cpp/run_sim.sh auto         # scripted: stand up, arm test, full dance (about 55 s)
#
# Keys in the MuJoCo window (keyboard gamepad, see simulate/src/main.cc):
#   1  LT + up         Passive -> FixStand (robot hangs on the elastic band)
#   8  (x5)            lower the band 0.5 m: the feet touch the floor
#   2  RB + X          FixStand -> Velocity (the walking policy holds it up)
#   9                  elastic band off
#   3  LT(2s) + up     Velocity -> Mimic_Schuhplattler_ArmTest (7 s, arms only, then back to Velocity)
#   4  LT(2s) + right  Velocity -> Mimic_Schuhplattler (2 s intro + dance, 16.96 s, then back to Velocity)
#   0  LT + B          any state -> Passive (motors damped: the robot falls, like the e-stop)
#   Backspace          reset the simulation
# Keys 3 and 4 hold "LT" for 2.6 s before pressing the second button, like the real pad.
HERE="$(cd "$(dirname "$0")" && pwd)"
SIM_DIR="$HERE/simulate/build"
CTRL_DIR="$HERE/deploy/robots/g1_29dof/build"
mkdir -p "$HERE/logs"

# leftovers of an earlier run (e.g. the terminal was closed) would send a second set of motor commands
for pid in $(pgrep -x g1_ctrl); do  # never touch a controller that drives the real robot
  if ! tr '\0' ' ' < /proc/$pid/cmdline | grep -q -- "--network lo"; then
    echo "g1_ctrl ($pid) is running on the REAL robot: stop it there first (L2 + B, then Ctrl+C)"; exit 1
  fi
done
for pid in $(pgrep -x unitree_mujoco) $(pgrep -x g1_ctrl); do
  case "$(readlink /proc/$pid/cwd)" in "$SIM_DIR"|"$CTRL_DIR") echo "stopping leftover $(cat /proc/$pid/comm) ($pid)"; kill $pid ;; esac
done
sleep 0.5

SCRIPT=""
[ "$1" = "auto" ] && SCRIPT="3:1,6:8x5,9:2,12:9,16:3,30:4"
cd "$SIM_DIR" && KB_LOG=${KB_LOG:-} KB_SCRIPT="$SCRIPT" ./unitree_mujoco > "$HERE/logs/sim_last.log" 2>&1 &
SIM=$!
sleep 2
cd "$CTRL_DIR" && ./g1_ctrl --network lo > "$HERE/logs/ctrl_last.log" 2>&1 &
CTRL=$!
trap 'kill $SIM $CTRL $TAIL 2>/dev/null' INT TERM EXIT
echo "MuJoCo window is open: click on it once (so it gets the keys), then press 1, 8 x5, 2, 9, 3, 4 (see the top of this file)."
echo "Ctrl+C here closes both programs. State changes:"
tail -n +1 -f --pid=$SIM "$HERE/logs/ctrl_last.log" "$HERE/logs/sim_last.log" | grep --line-buffered -E "FSM|keyboard pad|\[key\]|rror" &
TAIL=$!
if [ "$1" = "auto" ]; then sleep 55; else wait $SIM; fi
