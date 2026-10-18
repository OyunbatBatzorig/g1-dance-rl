# G1 Dance RL: does learning more dances make a humanoid more robust?

A two-week experiment teaching a simulated **Unitree G1 (29 DoF)** three dances with reinforcement learning in
**Isaac Lab**, then checking robustness under randomized physics and after transfer to **MuJoCo** (sim-to-sim).

| Dance | Source of the reference motion |
|---|---|
| Dance-102 | Unitree motion capture (from [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)) |
| Gangnam Style | Unitree motion capture (from unitree_rl_lab), first frames trimmed |
| Schuhplattler (Bavarian folk dance) | Own retarget from a tutorial video: video → [GVHMR](https://github.com/zju3dv/GVHMR) → [GMR](https://github.com/YanjieZe/GMR) → ground-contact, ending and balance fixes (`tools/`) |

Trained policies and the Schuhplattler reference motion are on Hugging Face: **[huggingface.co/Oyunbat26/g1-dance-rl](https://huggingface.co/Oyunbat26/g1-dance-rl)**

## Question and answer

**Does a policy that learns more motions become more robust?** In this experiment: **not clearly, but it did not
get worse either.** The biggest factor was the **quality of the reference motion**.

| Policy | Dances | Iterations | Action | MuJoCo: % of 64 randomized runs that finish the dance |
|---|---|---|---|---|
| Policy 1 | Dance-102 | 5,000 | residual | Dance-102 92% |
| Policy 2 | + Gangnam Style | 6,000 | residual | Dance-102 97%, Gangnam 97% |
| Policy 3 | + Schuhplattler (first reference) | 10,000 | residual | Dance-102 97%, Gangnam 100%, Schuhplattler 36% |
| Policy 3 full | same 3 dances, cleaned Schuhplattler (v2) | 10,000 | full | Dance-102 95%, Gangnam 95%, Schuhplattler 70% |
| Schuhplattler only | Schuhplattler (first reference) | 3,000 | full | 91–95% |
| Schuhplattler only | Schuhplattler (first reference) | 3,000 | residual | 0% |
| Schuhplattler only | Schuhplattler v2 (cleaned) | 3,000 | full | 88–95% |

* At **equal training** (the 5,000-iteration checkpoints) the single-dance policy was as robust or slightly
  better: Isaac Lab, Dance-102 with 1.5× wider randomization: P1 96.9%, P2 87.5%, P3 92.2%. Multi-dance policies
  mostly need more training.
* Robustness inside the training randomization was ~100% for Dance-102 and Gangnam for every policy; differences
  only show up at 1.5× wider ranges or in MuJoCo.
* On a poor reference (Schuhplattler, first retarget) the **residual** policy could not recover, the **full**
  action policy could. Cleaning the reference (balance fix, `tools/balance_fix.py`) plus the full action raised the
  3-dance policy from 36% to 70% on Schuhplattler.

**Limits:** one seed per policy, three dances, 3–10k iterations, 64 trials per test (about ±4%), simulation only
(no hardware). Differences of a few percent between Policies 1–3 may flip with another seed.

Full numbers: [`results/RESULTS.md`](results/RESULTS.md).

## Method

* **Base:** unitree_rl_lab mimic (BeyondMimic-style) tracking task for the G1, PPO (rsl-rl), 4096 envs,
  200 Hz physics, 50 Hz policy. Several dances in one policy via `MultiMotionCommand` (adaptive start-time
  sampling per clip).
* **Residual action (I-CTRL style):** `q_des = q_ref[k+1] + clip(a · scale, ±1.0 rad)`, with the reference joint
  velocity as PD velocity target. **Full action** (Unitree default): `q_des = q_default + scale · a`.
* **Domain randomization "DR v2"** (after I-CTRL Table II): Kp/Kd ×0.75–1.25, every link mass ×0.7–1.3, friction
  0.3–2.0 (static), actuation latency 0–20 ms, sensor noise (angular velocity ±0.2, joint position ±0.05, joint velocity
  ±1.0), random pushes from the original task.
* **Robustness tests:**
  * Isaac Lab Monte Carlo (`scripts/robust/robust_eval.py`): many robots, each with random dynamics from the
    training ranges (or ×1.5 wider), plus one-factor sweeps (mass, gains, friction, delay).
  * MuJoCo sim-to-sim (`scripts/sim2sim/sim2sim_dance.py`): same observation, action and PD as training, Unitree
    G1 MJCF, 64 runs with the training randomization. A run fails on the training termination rules.

## Repository layout

```
tasks/schuhplattler/            Isaac Lab tasks (drop into unitree_rl_lab, see below)
  tracking_env_cfg.py             base tracking task (adapted from unitree_rl_lab Dance-102)
  residual/residual_action.py     bounded residual joint-position action (+ full action mode, latency)
  residual/multi_motion.py        several motion clips in one policy
  residual/residual_env_cfg.py    DR v2 and all task variants
  residual/__init__.py            gym registrations (Unitree-G1-29dof-Mimic-Schuhplattler-*)
scripts/eval/day4_eval.py       play one policy on one clip (window, headless log, --replay, --video, --stress)
scripts/robust/                 Isaac Lab Monte Carlo / sweep robustness tests and tables
scripts/sim2sim/                MuJoCo sim-to-sim player and Monte Carlo test
tools/                          reference-motion tools: GMR retarget, CSV export, grounding, ending, arm copy,
                                balance fix (IK), motion and ZMP checks
results/                        result tables (Isaac Lab) and MuJoCo JSON results
```

## Setup

Tested with Isaac Sim 5.1, Isaac Lab 2.3, rsl-rl-lib 5.0.1, Python 3.11, RTX 3090.

1. Install [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) and check that its
   `Unitree-G1-29dof-Mimic-Dance-102` task trains.
2. Copy the tasks into it:
   ```bash
   cp -r tasks/schuhplattler <unitree_rl_lab>/source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/
   ```
3. Put the motion files (`.npz`, 50 Hz, made with unitree_rl_lab `scripts/mimic/csv_to_npz.py`) in a workspace
   folder, by default `~/projects/g1_pilot` (override with `G1_DANCE_DATA`):
   ```
   day6/dance_102.npz            from unitree_rl_lab G1_Take_102.bvh_60hz.csv
   day6/gangnam_style_trim.npz   from unitree_rl_lab G1_gangnam_style_V01.bvh_60hz.csv, first 2 frames removed
   day2/schuhplattler_gmr_end_grounded.npz   Schuhplattler, first reference   (Hugging Face)
   day2/schuhplattler_gmr_end_bal2.npz       Schuhplattler, v2 reference      (Hugging Face)
   ```
   The scripts in `scripts/` also assume this workspace (`~/projects/g1_pilot`) for their outputs.
4. For MuJoCo: a separate env with `mujoco`, `torch`, `numpy` (and `imageio[ffmpeg]` for `--record`), plus the
   G1 MJCF from [unitree_ros](https://github.com/unitreerobotics/unitree_ros) (`--xml`).

## Usage

```bash
# train (from the unitree_rl_lab root)
python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Mimic-Schuhplattler-Residual-DR-D102-Wide-Multi3 \
    --max_iterations 10000 --run_name d102_p3

# watch one clip in Isaac Lab (window); --headless saves a log, --replay plays it back at true speed
python <repo>/scripts/eval/day4_eval.py --task <task> --checkpoint logs/rsl_rl/g1_schuhplattler_residual/<run> --clip gangnam

# Isaac Lab Monte Carlo robustness (training DR and 1.5x wider) + sweeps
bash <repo>/scripts/robust/robust_all.sh logs/rsl_rl/g1_schuhplattler_residual/<run> p3 <task> "dance_102 gangnam schuhplattler"

# MuJoCo sim-to-sim: viewer (Space = start/pause) or 64 Monte Carlo runs
python <repo>/scripts/sim2sim/sim2sim_dance.py --checkpoint <run>/model_9999.pt --motion <npz> --action wide --viewer
python <repo>/scripts/sim2sim/sim2sim_dance.py --checkpoint <run>/model_9999.pt --motion <npz> --action wide --runs 64 --name p3_d102
```

`--action wide` for the residual tasks (`...-Residual-DR-...-Wide...`), `--action mimic` for the full-action tasks
(`...-MimicAction-...`).

| Task id (prefix `Unitree-G1-29dof-Mimic-Schuhplattler-`) | Policy |
|---|---|
| `Residual-DR-D102-Wide` | Policy 1 (Dance-102) |
| `Residual-DR-D102-Wide-Multi2` | Policy 2 (+ Gangnam) |
| `Residual-DR-D102-Wide-Multi3` | Policy 3 (+ Schuhplattler, first reference) |
| `MimicAction-DR-D102-Multi3Bal` | Policy 3 full (Schuhplattler v2) |
| `MimicAction-DR-Sch` / `Residual-DR-Sch-Wide` | Schuhplattler only, full / residual |
| `MimicAction-DR-SchBal2` | Schuhplattler v2 only, full |

## Credits and licenses

This code is released under the **Apache License 2.0** (see `LICENSE` and `NOTICE`). It builds on:

* [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) (Apache-2.0): tasks, PPO setup,
  Dance-102 and Gangnam Style motion capture
* [Isaac Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause)
* [unitree_ros](https://github.com/unitreerobotics/unitree_ros) (BSD-3-Clause): G1 URDF / MJCF
* [GMR: General Motion Retargeting](https://github.com/YanjieZe/GMR) (MIT), "Retargeting Matters" paper
* [GVHMR](https://github.com/zju3dv/GVHMR) (non-commercial research license) and SMPL-X body models
  (non-commercial, not redistributed here)
* [I-CTRL](https://arxiv.org/abs/2405.08726): "Imitation to Control Humanoid Robots Through Constrained Reinforcement Learning"
* Schuhplattler reference video: "Schuhplatteln lernen (Anleitung): Häusei / Häuslratz" by iPlattl
  (only used to extract the motion, no footage included)

The Schuhplattler motion data is derived from GVHMR output and is shared on Hugging Face for **non-commercial
use only** (CC BY-NC 4.0).
