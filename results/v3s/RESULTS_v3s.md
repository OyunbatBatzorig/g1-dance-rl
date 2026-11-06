# Schuhplattler v3s: training, robustness and sim-to-real (Oct 30 – Nov 2, 2026)

Reference **v3s** = v2 reference + 52 manual edits in the reference editor (hand/foot poses by IK, local speed
changes), then the local speed changes replaced by one smooth tempo curve (`tools/tempo_smooth.py`, hand speed
≤ 3.5 m/s). 15.0 s at 50 Hz. Full (mimic) action for every policy below.

| Policy | Task (prefix `Unitree-G1-29dof-Mimic-Schuhplattler-`) | Iterations | Randomization |
|---|---|---|---|
| plus | `MimicAction-DRPlus-SchV3s` | 30,000 (from scratch, 17 h) | DR v2 + DR+ |
| **floor** | `MimicAction-DRFloor-SchV3s` | +5,000 from plus model_29999 (3 h) | DR v2 + DR+ + floor mix, friction 0.2–2.0 |

**DR+** adds sim-to-real extras on top of DR v2:
- joint zero offset ±0.03 rad
- torso centre-of-mass shift ±5 cm
- motor armature ×0.8–1.2
- IMU orientation noise ±0.08
- uneven floor, 0–1 cm

**Floor mix:** each robot gets one floor tile:
- flat 30%
- rough 0–1 cm 30%
- stone-like 0–2.5 cm 20%
- pyramid slope ≤ 5% grade, 0.5 m top, 20%

## MuJoCo sim-to-sim, 512 runs per test (`mujoco_n512/`, 95% Wilson CI)

"Finished" uses the training terminations. A "real fall" re-plays every terminated run to the end without
stopping, and counts it only when the pelvis drops below 0.45 m or the torso tilts more than 60°
(`scripts/sim2sim/classify_falls.py`).

| Test | floor finished | floor real falls | plus finished | plus real falls |
|---|---|---|---|---|
| training DR | 97.9% [96.2–98.8] | 0.0% [0.0–0.7] | 98.4% [96.9–99.2] | 0.2% [0.0–1.1] |
| DR ×1.5 | 93.4% [90.9–95.2] | 1.4% | 93.2% [90.6–95.0] | 1.4% |
| friction 0.3 | 76.8% [72.9–80.2] | 18.0% | 80.7% [77.0–83.9] | 17.2% |
| friction 0.25 | **18.4%** [15.2–21.9] | **78.3%** | 7.4% [5.5–10.0] | 91.0% |

On a flat floor almost every termination is a tracking deviation, not a fall:
- 16 of 19 happen at 8.50–8.56 s. In this heel-slap the reference lifts the right foot from 0.24 m to 0.50 m,
  and the robot lifts it less than that.
- The robot stays upright and is back within 0.25 m of the reference by the end of the clip.

The two policies are equal on the flat floor. The floor policy tracks more closely (body error 4.4 vs 4.7 cm)
and falls less on a slippery floor.

Uneven floors could not be judged in MuJoCo:
- The G1 MJCF foot is four 5 mm spheres.
- A zero-height heightfield gives 0–58% (depending on CCD settings) and a grid of 10 cm boxes gives 17%,
  against 98% on the plane.
- These numbers measure the contact model, not the policy, so floor types were evaluated in Isaac Lab instead.

## Isaac Lab, one floor type at a time, 512 robots (`isaac_floor/`)

DR v2 with friction 0.3–2.0. "ee off" removes the hand and foot height termination, so it counts torso-height
and tilt failures only (close to real falls).

| Floor | floor: terminated | floor: ee off | plus: terminated | plus: ee off |
|---|---|---|---|---|
| flat | 0.2% | 0.2% | 0.0% | 0.0% |
| rough 0–1 cm | 0.6% | 0.2% | 0.6% | 0.0% |
| stone 0–2.5 cm | 2.3% | 0.0% | 3.7% | 0.2% |
| slope 0–5% | **11.5%** | 0.2% | 19.1% | 0.2% |

Joint RMSE on every floor: floor 0.191–0.193 rad, plus 0.207–0.208 rad.

On slopes the reference foot height stays relative to a flat floor, so part of the slope terminations come
from geometry, not tracking.

## Deploy logic in MuJoCo (`deploy_sim2sim/`, `scripts/deploy/deploy_sim2sim.py`)

A Python copy of unitree_rl_lab's C++ `State_Mimic`:
- CSV loader with `round` + blend
- joint velocity by forward difference
- torso orientation from the pelvis IMU and the waist joints, `init_quat` yaw alignment
- per-motor gains, ONNX policy

Same DR and seeds as `sim2sim_dance.py`, 512 runs each.

| Test | Finished | Real falls |
|---|---|---|
| ONNX + deploy observations, start on the reference | 99.2% | 0.0% |
| start from standing, no intro | 98.8% | 0.4% |
| **start from standing, 2 s intro blend** | **99.4%** | **0.0%** |
| playback 0.9× (`fps` 45) | 96.9% | 1.8% |
| playback 0.75× | 57.8% | 23.2% |
| playback 0.5× | 0.0% | 99.8% |
| arm smoke test (arms only, 7 s) | 100% | 0.0% |
| part: first 1.66 s + 2 s calm return | 99.8% | 0.2% |
| part: first 11.04 s + return | 0.0% | 29.3% (dropped) |

Playing the dance slower is not safer:
- One foot carries the robot 67% of the time.
- In single support the centre of mass is above the stance foot in only 63% of the frames
  (`scripts/deploy/balance_check.py`).
- So the dance relies on its own dynamics, and the policy was only trained at 1×.

ONNX vs PyTorch: max |Δaction| 7.6e-6.

## Real Unitree G1 (Nov 2, 2026)

Setup: the real `g1_ctrl` (unitree_rl_lab deploy, patched, see `sim_cpp/`), robot on a gantry, wired Ethernet.

Order: FixStand → Velocity → arm smoke test → part → full dance.

The **full dance ran 6 of 6 times**, each 16.96 s, and handed back to the Velocity policy every time.

A loose Ethernet cable later triggered the lowstate-timeout fall-back to Passive while the robot stood in
Velocity. Fix the cable: if it drops mid-dance, that Passive command cannot reach the robot.
