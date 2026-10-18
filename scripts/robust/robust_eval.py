"""Robustness evaluation of a residual policy with many robots at once (2026-10-16).

day4_eval.py --stress runs ONE robot at fixed worst-case corners. This script runs N robots in parallel,
all starting the dance at frame 0, and counts how many finish the whole clip without a termination.

  --mode mc     Monte Carlo: every robot gets its own random dynamics from the task's training DR
                (DR v2: Kp/Kd x0.75-1.25, link mass x0.7-1.3, friction 0.3-2.0, 0-20 ms latency,
                sensor noise, torso CoM shift, joint calibration offset). --widen k stretches the
                gains/mass ranges and the sensor noise k times around nominal (k > 1 = beyond training).
  --mode sweep  one factor (--factor mass|gains|friction|delay) is set to each of --values,
                --per_value robots per value; everything else is random training DR (--others dr)
                or switched off (--others nominal). Shows where that factor breaks the policy.

Run from ~/projects/unitree_rl_lab with env_isaaclab active, e.g.
  python ~/projects/g1_pilot/robust/robust_eval.py --task <task> --checkpoint logs/rsl_rl/g1_schuhplattler_residual/<run> \
      --clip dance_102 --mode mc --num_envs 256 --name p2__dance_102__mc --headless
Output: ~/projects/g1_pilot/robust/<name>/results.json and summary.txt (robust_table.py makes the tables).
"""

import argparse
import glob
import os
import re
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True, help="model_N.pt, or a run folder (its newest model_N.pt is used)")
parser.add_argument("--clip", default=None, help="dance name prefix or index (multi-clip tasks)")
parser.add_argument("--mode", choices=("mc", "sweep"), default="mc")
parser.add_argument("--num_envs", type=int, default=256, help="mc: number of robots")
parser.add_argument("--widen", type=float, default=1.0, help="mc: gains/mass range and sensor noise x this")
parser.add_argument("--factor", choices=("mass", "gains", "friction", "delay"), default="mass")
parser.add_argument("--values", default=None, help="sweep: comma list (mass/gains: scale, friction: mu, delay: ms)")
parser.add_argument("--per_value", type=int, default=32, help="sweep: robots per value")
parser.add_argument("--others", choices=("dr", "nominal"), default="dr", help="sweep: the other factors")
parser.add_argument("--push", action="store_true", help="keep the task's random pushes on")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--name", required=True, help="output folder name")
parser.add_argument("--out", default=os.path.expanduser("~/projects/g1_pilot/robust"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
sys.stdout.reconfigure(line_buffering=True)

DEFAULT_VALUES = {"mass": "0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6",
                  "gains": "0.5,0.6,0.7,0.75,0.8,0.9,1.0,1.1,1.25,1.4",
                  "friction": "0.1,0.2,0.3,0.5,0.8,1.0,1.5,2.0",
                  "delay": "0,5,10,15,20"}
values = [float(v) for v in (args.values or DEFAULT_VALUES[args.factor]).split(",")]
if args.mode == "sweep" and args.factor == "delay":
    assert all(0 <= v <= 20 and v % 5 == 0 for v in values), "delay: multiples of 5 ms, 0-20 (residual action holds at most one step)"
num_envs = args.num_envs if args.mode == "mc" else args.per_value * len(values)

app = AppLauncher(args).app

import json  # noqa: E402
from importlib.metadata import version  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
import unitree_rl_lab.tasks  # noqa: E402,F401
from isaaclab.actuators import ImplicitActuator  # noqa: E402
from isaaclab.utils.noise import AdditiveUniformNoiseCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402


def resolve_checkpoint(p):
    p = os.path.expanduser(p)
    if os.path.isdir(p):
        files = glob.glob(os.path.join(p, "model_*.pt"))
        if not files:
            raise SystemExit(f"no model_*.pt in {p}")
        return max(files, key=lambda f: int(re.findall(r"model_(\d+)\.pt", f)[0]))
    if not os.path.isfile(p):
        raise SystemExit(f"checkpoint not found: {p}")
    return p


ckpt = resolve_checkpoint(args.checkpoint)
out = os.path.join(args.out, args.name)
os.makedirs(out, exist_ok=True)

# ---- config: the task's training DR, every robot starts the clip at frame 0 -------------------
cfg = parse_env_cfg(args.task, num_envs=num_envs)
cfg.seed = args.seed
cfg.commands.motion.pose_range = {}
cfg.commands.motion.velocity_range = {}
cfg.commands.motion.joint_position_range = (0.0, 0.0)
cfg.episode_length_s = 3600.0  # never time out; the loop stops at the clip's last frame
if not args.push and hasattr(cfg.events, "push_robot"):
    cfg.events.push_robot = None
act_cfg = cfg.actions.JointPositionAction
assert hasattr(act_cfg, "delay_substeps_range"), "needs the residual action with latency (residual_action.py)"

if args.mode == "mc" and args.widen != 1.0:
    k = args.widen
    if getattr(cfg.events, "actuator_gains", None) is not None:
        cfg.events.actuator_gains.params["stiffness_distribution_params"] = (1 - 0.25 * k, 1 + 0.25 * k)
        cfg.events.actuator_gains.params["damping_distribution_params"] = (1 - 0.25 * k, 1 + 0.25 * k)
    if getattr(cfg.events, "link_mass", None) is not None:
        cfg.events.link_mass.params["mass_distribution_params"] = (max(0.1, 1 - 0.3 * k), 1 + 0.3 * k)
    pol = cfg.observations.policy
    for nm in dir(pol):
        t = getattr(pol, nm, None)
        if isinstance(getattr(t, "noise", None), AdditiveUniformNoiseCfg):
            t.noise = AdditiveUniformNoiseCfg(n_min=t.noise.n_min * k, n_max=t.noise.n_max * k)

if args.mode == "sweep":
    if args.others == "nominal":
        for nm in ("physics_material", "base_com", "add_joint_default_pos", "actuator_gains", "link_mass"):
            if hasattr(cfg.events, nm):
                setattr(cfg.events, nm, None)
        cfg.observations.policy.enable_corruption = False
        act_cfg.delay_substeps_range = (0, 0)
    # the swept factor is written per robot below, not drawn by its DR event
    swept_event = {"mass": "link_mass", "gains": "actuator_gains", "friction": "physics_material"}.get(args.factor)
    if swept_event and hasattr(cfg.events, swept_event):
        setattr(cfg.events, swept_event, None)
    if args.factor == "delay":
        act_cfg.delay_substeps_range = (0, 4)  # keeps the latency path on; values are set per robot below

env = gym.make(args.task, cfg=cfg)
u = env.unwrapped
robot = u.scene["robot"]
cmd = u.command_manager.get_term("motion")
term = u.action_manager.get_term("JointPositionAction")
assert hasattr(cmd, "set_clip_start"), "this script needs a MultiMotionCommand task (DR v2 / Wide tasks)"
hits = [i for i, n in enumerate(cmd.clip_names) if args.clip is not None and n.startswith(str(args.clip))]
clip = hits[0] if hits else int(args.clip or 0)
clip_name = cmd.clip_names[clip]
start = int(cmd.clip_start[clip])
T = int(cmd.motion.clip_lengths[clip])
step_dt = u.step_dt


def _start_at_zero(env_ids):
    cmd.set_clip_start(env_ids, clip, 0)


cmd._adaptive_sampling = _start_at_zero  # must be patched before the wrapper resets the env

def set_mass_scale(ids, v):
    """every link mass (and inertia) = default x v, like randomize_rigid_body_mass with a fixed scale"""
    ic = ids.cpu()
    masses = robot.root_physx_view.get_masses()
    masses[ic] = robot.data.default_mass[ic] * v
    robot.root_physx_view.set_masses(masses, ic)
    inertias = robot.root_physx_view.get_inertias()
    inertias[ic] = robot.data.default_inertia[ic] * v
    robot.root_physx_view.set_inertias(inertias, ic)


def set_gain_scale(ids, v):
    """Kp and Kd of every joint = default x v, like randomize_actuator_gains with a fixed scale"""
    for act in robot.actuators.values():
        ji = act.joint_indices
        kp = robot.data.default_joint_stiffness[ids][:, ji] * v
        kd = robot.data.default_joint_damping[ids][:, ji] * v
        act.stiffness[ids] = kp
        act.damping[ids] = kd
        if isinstance(act, ImplicitActuator):
            robot.write_joint_stiffness_to_sim(kp, joint_ids=ji, env_ids=ids)
            robot.write_joint_damping_to_sim(kd, joint_ids=ji, env_ids=ids)


def gain_scale_per_robot():
    r = [act.stiffness / robot.data.default_joint_stiffness[:, act.joint_indices].clamp(min=1e-6)
         for act in robot.actuators.values()]
    return torch.cat(r, 1).mean(1)


# ---- sweep: write the swept factor per robot --------------------------------------------------
group = torch.arange(num_envs, device=u.device) // max(1, args.per_value)  # sweep: value index per robot
if args.mode == "sweep" and args.factor in ("mass", "gains", "friction"):
    for gi, v in enumerate(values):
        ids = torch.nonzero(group == gi).flatten()
        if args.factor == "mass":
            set_mass_scale(ids, v)
        elif args.factor == "gains":
            set_gain_scale(ids, v)
        else:
            mats = robot.root_physx_view.get_material_properties()
            ic = ids.cpu()
            mats[ic, :, 0] = v
            mats[ic, :, 1] = v
            mats[ic, :, 2] = 0.0
            robot.root_physx_view.set_material_properties(mats, ic)

agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))
venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)  # resets every robot to frame 0

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

runner = OnPolicyRunner(venv, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(ckpt)
policy = runner.get_inference_policy(device=u.device)

if args.mode == "sweep" and args.factor == "delay":  # after the reset, which draws a new delay per robot
    for gi, v in enumerate(values):
        term._delay[group == gi] = int(round(v / 5.0))

# ---- what each robot actually got -------------------------------------------------------------
dm = robot.data.default_mass
params = {
    "mass_scale": (robot.root_physx_view.get_masses().sum(1) / dm.sum(1)).numpy(),
    "gains_scale": gain_scale_per_robot().cpu().numpy(),
    "friction": robot.root_physx_view.get_material_properties()[:, :, 0].mean(1).numpy(),
    "delay_ms": (term._delay.float() * 5.0).cpu().numpy(),
}

# ---- run ----------------------------------------------------------------------------------------
cmd.time_steps[:] = start - 1  # same fix as day4_eval: fill frame 0 before the first step
cmd._update_command()
obs = venv.get_observations()
if isinstance(obs, tuple):
    obs = obs[0]

alive = torch.ones(num_envs, dtype=torch.bool, device=u.device)
fall_step = torch.full((num_envs,), T - 1, dtype=torch.long, device=u.device)
cause = [""] * num_envs
sq_err = torch.zeros(num_envs, device=u.device)
n_err = torch.zeros(num_envs, device=u.device)
tm = u.termination_manager
print(f"[robust] task={args.task} clip={clip_name} ({T} frames, {(T - 1) * step_dt:.2f} s) mode={args.mode} "
      f"robots={num_envs} policy={ckpt}")
with torch.inference_mode():
    for k in range(T - 1):
        if not app.is_running():
            break
        obs, _, dones, _ = venv.step(policy(obs))
        hit = dones.bool() & alive
        if hit.any():
            for i in torch.nonzero(hit).flatten().tolist():
                fall_step[i] = k + 1
                cause[i] = ",".join(n for n in tm.active_terms if bool(tm.get_term(n)[i])) or "terminated"
            alive &= ~hit
        e = (robot.data.joint_pos - cmd.joint_pos) ** 2
        sq_err += torch.where(alive, e.mean(1), torch.zeros_like(sq_err))
        n_err += alive.float()
        if k % 250 == 0:
            print(f"[robust] t={(k + 1) * step_dt:5.1f}s  still dancing {int(alive.sum())}/{num_envs}")
        if not alive.any():
            break

ok = alive.cpu().numpy()
fall_s = (fall_step.float() * step_dt).cpu().numpy()
rmse = torch.sqrt(sq_err / n_err.clamp(min=1)).cpu().numpy()
motion_s = (T - 1) * step_dt

res = {"task": args.task, "policy": ckpt, "clip": clip_name, "mode": args.mode, "widen": args.widen,
       "factor": args.factor if args.mode == "sweep" else None, "others": args.others if args.mode == "sweep" else None,
       "push": args.push, "seed": args.seed, "num_envs": num_envs, "motion_s": round(motion_s, 2),
       "success_pct": round(100 * float(ok.mean()), 1),
       "completed_pct_mean": round(100 * float((fall_s / motion_s).mean()), 1),
       "joint_rmse_mean_rad": round(float(rmse.mean()), 4),
       "per_robot": {"ok": ok.tolist(), "fall_s": fall_s.round(2).tolist(), "cause": cause,
                     "rmse": rmse.round(4).tolist(), **{p: v.round(4).tolist() for p, v in params.items()}}}
lines = [f"{args.name}: {clip_name}, {args.mode}, {num_envs} robots, {res['policy']}",
         f"success (whole dance, no termination): {res['success_pct']}%   mean completed {res['completed_pct_mean']}%"
         f"   joint rmse {res['joint_rmse_mean_rad']} rad"]
if args.mode == "sweep":
    res["sweep"] = []
    lines.append(f"{args.factor:>9s}  success%  completed%  first falls at [s]")
    for gi, v in enumerate(values):
        m = (group == gi).cpu().numpy()
        f = np.sort(fall_s[m & ~ok])[:3]
        res["sweep"].append({"value": v, "success_pct": round(100 * float(ok[m].mean()), 1),
                             "completed_pct": round(100 * float((fall_s[m] / motion_s).mean()), 1)})
        lines.append(f"{v:9.2f}  {res['sweep'][-1]['success_pct']:7.1f}  {res['sweep'][-1]['completed_pct']:9.1f}   "
                     + " ".join(f"{x:.1f}" for x in f))
else:
    res["by_param"] = {}
    lines.append("success % by sampled parameter (thirds of the sampled range):")
    for p, v in params.items():
        lo, hi = float(v.min()), float(v.max())
        if hi - lo < 1e-6:
            continue
        edges = np.linspace(lo, hi, 4)
        cells = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (v >= a) & (v <= b)
            cells.append({"range": [round(a, 3), round(b, 3)], "n": int(m.sum()),
                          "success_pct": round(100 * float(ok[m].mean()), 1) if m.any() else None})
        res["by_param"][p] = cells
        lines.append(f"  {p:12s} " + "   ".join(f"[{c['range'][0]:.2f}-{c['range'][1]:.2f}] {c['success_pct']}% (n={c['n']})"
                                               for c in cells))
if (~ok).any():
    h = np.bincount(fall_s[~ok].astype(int))
    top = np.argsort(-h)[:5]
    res["fall_seconds_top"] = {f"{int(s)}-{int(s) + 1}s": int(h[s]) for s in top if h[s] > 0}
    lines.append("where in the dance robots fall (1 s bins, most first): "
                 + ", ".join(f"{a}: {b}" for a, b in res["fall_seconds_top"].items()))
    causes = {}
    for c, o in zip(cause, ok):
        if not o:
            causes[c] = causes.get(c, 0) + 1
    res["causes"] = causes
    lines.append(f"termination causes: {causes}")
json.dump(res, open(os.path.join(out, "results.json"), "w"))
open(os.path.join(out, "summary.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
env.close()
app.close()
