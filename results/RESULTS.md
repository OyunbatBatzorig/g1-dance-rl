# Results

All numbers: one training seed (42) per policy. "Finished" = the robot completed the whole clip without
triggering a training termination (torso height / hand or foot height off the reference by > 0.25 m, or torso
tilt off). With 64 trials one run is ~1.6%, so differences below ~4–5% are within noise.

## Policies

| Name | Task (prefix `Unitree-G1-29dof-Mimic-Schuhplattler-`) | Dances | Iterations | Action |
|---|---|---|---|---|
| P1 | `Residual-DR-D102-Wide` | Dance-102 | 5,000 | residual |
| P2 | `Residual-DR-D102-Wide-Multi2` | + Gangnam Style | 6,000 | residual |
| P3 | `Residual-DR-D102-Wide-Multi3` | + Schuhplattler (first reference) | 10,000 | residual |
| P3 full | `MimicAction-DR-D102-Multi3Bal` | Dance-102, Gangnam, Schuhplattler v2 | 10,000 | full |
| Sch full | `MimicAction-DR-Sch` | Schuhplattler (first reference) | 3,000 | full |
| Sch residual | `Residual-DR-Sch-Wide` | Schuhplattler (first reference) | 3,000 | residual |
| Sch v2 full | `MimicAction-DR-SchBal2` | Schuhplattler v2 | 3,000 | full |

All are trained from scratch with DR v2.

## Training (Isaac Lab, end of training)

| Policy | Mean episode length (steps) | Share of episodes reaching time-out | Body position error |
|---|---|---|---|
| P3 | 1167 | 0.64 | 0.104 |
| P3 full | ~1235–1300 | 0.73 | ~0.083–0.095 |
| Sch residual (3k) | 575 | 0.15 | 0.195 |
| Sch full (3k) | 969 | 0.37 | 0.144 |

## Isaac Lab Monte Carlo (final checkpoints)

Each robot gets random dynamics from the training ranges, or 1.5× wider gains / mass ranges.
Source: `isaac/p1_table.txt`, `p2_table.txt`, `p3_table.txt` (these also contain the one-factor sweeps).

| Dance | Test | P1 | P2 | P3 |
|---|---|---|---|---|
| Dance-102 | training DR | 99.6% | 100% | 100% |
| Dance-102 | 1.5× wider | 97.3% | 94.5% | 98.0% |
| Gangnam | training DR | – | 99.6% | 100% |
| Gangnam | 1.5× wider | – | 87.9% | 88.3% |
| Schuhplattler | training DR | – | – | 76.6% |
| Schuhplattler | 1.5× wider | – | – | 50.8% |

## Isaac Lab Monte Carlo: full-action and Schuhplattler-only policies (64 robots per test)

Source: `isaac/final_table.txt`.

| Dance | Test | P3 full | Sch full | Sch residual |
|---|---|---|---|---|
| Dance-102 | training DR | 100% | – | – |
| Dance-102 | 1.5× wider | 100% | – | – |
| Gangnam | training DR | 100% | – | – |
| Gangnam | 1.5× wider | 78.1% | – | – |
| Schuhplattler | training DR | 93.8% (v2 reference) | 89.1% | 20.3% |
| Schuhplattler | 1.5× wider | 70.3% (v2 reference) | 67.2% | 4.7% |

Sch full / Sch residual use the first Schuhplattler reference. Residual falls happen mostly at 1–3 s (first kick).

## Isaac Lab Monte Carlo at equal training (5,000-iteration checkpoints, 64 robots per test)

Source: `isaac/iso5k_table.txt`.

| Dance | Test | P1 | P2 | P3 |
|---|---|---|---|---|
| Dance-102 | training DR | 100% | 100% | 100% |
| Dance-102 | 1.5× wider | 96.9% | 87.5% | 92.2% |
| Gangnam | training DR | – | 100% | 100% |
| Gangnam | 1.5× wider | – | 78.1% | 79.7% |
| Schuhplattler | training DR | – | – | 4.7% (not learned yet at 5k) |

## MuJoCo sim-to-sim (64 runs with the training randomization)

Source: `mujoco/*.json` (`python ../scripts/sim2sim/table.py mujoco`).

| Policy | Dance-102 | Gangnam | Schuhplattler |
|---|---|---|---|
| P1 | 92% | – | – |
| P2 | 97% | 97% | – |
| P3 | 97% | 100% | 36% (first reference) |
| P3 full | 95% | 95% | 70% (v2 reference) |
| Sch full | – | – | 95% (seed 0), 91% (seed 1000) |
| Sch residual | – | – | 0% |
| Sch v2 full | – | – | 95% (seed 0), 88% (seed 1000) |

At 5,000 iterations (`mujoco/iso5k`, 128 runs): Dance-102 P1 95%, P2 90%, P3 84%; Gangnam P2 92%, P3 95%.

## Takeaways

1. More dances did not clearly improve robustness, and did not clearly hurt it.
2. At equal iterations the single-dance policy is as good or better: multi-dance policies need longer training.
3. Reference quality mattered most. With a poor reference the residual action fails (0–36%) while the full
   action still works (Schuhplattler alone: Isaac Lab 89%, MuJoCo 91–95%; residual 20% / 0%).
4. Policies trained with DR v2 transferred to MuJoCo with 92–100% success on the Unitree dances.

Note on Schuhplattler alone: with the full action, the cleaned v2 reference gave the same MuJoCo result as the first
reference (88–95% vs 91–95%). The reference fix mattered for the multi-dance policy (P3 36% → P3 full 70%), where the
first reference was the hardest clip.
