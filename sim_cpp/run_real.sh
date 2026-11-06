#!/bin/bash
# Run the Schuhplattler deploy (g1_ctrl from this folder) on the REAL G1 over Ethernet.
#   bash ~/projects/g1_pilot/sim_cpp/run_real.sh <network interface, e.g. enp3s0>
# The robot's own motion control must be off (debug mode) first -- see deploy/sim2real_runbook.md.
HERE="$(cd "$(dirname "$0")" && pwd)"
IF="$1"
[ -z "$IF" ] && { echo "usage: run_real.sh <interface>   (the wired interface on 192.168.123.x, see: ip -br addr)"; exit 1; }
ip -br addr show "$IF" >/dev/null 2>&1 || { echo "no interface $IF"; ip -br addr; exit 1; }
ip -br addr show "$IF" | grep -q "192.168.123." || { echo "warning: $IF has no 192.168.123.x address:"; ip -br addr show "$IF"; }
if pgrep -x unitree_mujoco >/dev/null || pgrep -x g1_ctrl >/dev/null; then
  echo "a simulator or another g1_ctrl is still running -- close it first:"; pgrep -ax "unitree_mujoco|g1_ctrl"; exit 1
fi
BLK=$(grep -A9 "^  Mimic_Schuhplattler:" "$HERE/deploy/robots/g1_29dof/config/config.yaml")
TE=$(echo "$BLK" | grep time_end | awk '{print $2}')
MF=$(basename "$(echo "$BLK" | grep motion_file | awk '{print $2}')")
echo "Schuhplattler: $MF until $TE s"
# allowed: the whole dance, or the 1.66 s part with its calm 2 s return; both end in the Velocity stance.
# A shorter time_end on the whole dance would hand over mid-motion (Velocity snaps the arms down).
case "$MF:$TE" in
  schuhplattler_gmr_v3s_50hz_intro2.csv:16.96|schuhplattler_v3s_part01.66_50hz.csv:6.66) ;;
  *) echo "not a safe setting for the robot: bash set_dance.sh full  (or: part)"; exit 1 ;;
esac
cat <<'TXT'
Checklist: gantry on, feet off the floor | remote in the safety person's hand, L2 + B tried | battery charged |
           floor dry (not polished) | 2 m clear around | video recording | robot's own control off (debug mode)
Remote: L2+up FixStand | R1+X Velocity | L2(2s)+up ArmTest | L2(2s)+right Schuhplattler | L2+B Passive (stop)
TXT
read -r -p "Type yes to start g1_ctrl on $IF: " ok
[ "$ok" = "yes" ] || exit 1
mkdir -p "$HERE/logs"
cd "$HERE/deploy/robots/g1_29dof/build" && PAD_LOG=1 ./g1_ctrl --network "$IF" 2>&1 | tee "$HERE/logs/real_$(date +%Y%m%d_%H%M%S).log"
