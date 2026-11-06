#!/bin/bash
# Build the C++ sim2sim / sim2real setup used for the real G1 test (2026-11-02):
#   simulate/  = unitree_mujoco's simulator + a keyboard "gamepad" (number keys play the FSM button combos)
#   deploy/    = unitree_rl_lab's g1_ctrl + the Schuhplattler states + a remote-button log (PAD_LOG=1)
# Both are copies next to this script; the upstream repos are not changed.
#   bash setup.sh <unitree_mujoco repo> <unitree_rl_lab repo> <deploy package folder (schuhplattler_v3s)>
# Tested with unitree_rl_lab 4960b84 and unitree_sdk2 installed in /opt/unitree_robotics.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
MJ=${1:?unitree_mujoco repo}; RL=${2:?unitree_rl_lab repo}; PKG=${3:?deploy package folder}

# simulator
rm -rf "$HERE/simulate"; mkdir -p "$HERE/simulate"
cp -r "$MJ/simulate/src" "$MJ/simulate/CMakeLists.txt" "$MJ/simulate/config.yaml" "$HERE/simulate/"
ln -sfn "$(readlink -f "$MJ/simulate/mujoco")" "$HERE/simulate/mujoco"
ln -sfn "$(readlink -f "$MJ/unitree_robots")" "$HERE/unitree_robots"
(cd "$HERE" && patch -p1 < unitree_mujoco_keyboard.patch)
sed -i 's/^robot: .*/robot: "g1"/; s/^robot_scene: .*/robot_scene: "scene_29dof.xml"/; s/^domain_id: .*/domain_id: 0/;
        s/^use_joystick: .*/use_joystick: 1/; s/^joystick_type: .*/joystick_type: "keyboard"/; s/^interface: .*/interface: "lo"/;
        s/^enable_elastic_band: .*/enable_elastic_band: 1/' "$HERE/simulate/config.yaml"

# g1_ctrl
rm -rf "$HERE/deploy"; mkdir -p "$HERE/deploy/robots"
ln -sfn "$(readlink -f "$RL/deploy/include")" "$HERE/deploy/include"
ln -sfn "$(readlink -f "$RL/deploy/thirdparty")" "$HERE/deploy/thirdparty"
rsync -a --exclude build "$RL/deploy/robots/g1_29dof" "$HERE/deploy/robots/"
(cd "$HERE/deploy/robots/g1_29dof" && patch -p1 < "$HERE/g1_ctrl.patch")
cp -r "$PKG" "$HERE/deploy/robots/g1_29dof/config/policy/mimic/schuhplattler_v3s"

# build both
(cd "$HERE/simulate" && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null && make -j"$(nproc)" unitree_mujoco)
(cd "$HERE/deploy/robots/g1_29dof" && mkdir -p build && cd build && cmake .. > /dev/null && make -j"$(nproc)")
echo "done: bash $HERE/run_sim.sh   (see README.md)"
