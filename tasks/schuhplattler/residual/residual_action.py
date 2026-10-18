"""Bounded residual joint-position action (Day 4, I-CTRL style).

    q_des = q_ref[k + lookahead] + clip(residual_scale * a, -delta_j, +delta_j)

q_ref comes from the motion command (the same npz the rewards track), so the policy only learns a small
correction on top of the reference. With a = 0 this is exactly the Day 3 PD baseline (target = next frame).

Optional velocity feed-forward (ref_velocity_ff): the PD also gets the reference joint velocity as its
velocity target, so tau = Kp (q_des - q) + Kd (qd_ref - qd) instead of Kd (0 - qd). Without it the soft
G1 PD (Kd/Kp = 0.064 s) lags a moving reference by ~0.064 * qd, more than the arm bound on fast moves.
The G1 SDK takes a dq target too, so the same command exists on the real robot.

ref_base=False turns the term into the original mimic action (Unitree / BeyondMimic):
q_des = default + scale * a, no reference in the action and no bound. Used only as a control experiment,
so that it gets exactly the same latency and observation pipeline as the residual tasks.

delta_j is residual_limit for every joint, or per joint group when residual_limit_groups is set
(I-CTRL: arms 10 deg, legs and torso 30 deg).

Optional actuation latency (DR v2): each env gets a delay of d physics sub-steps (sampled per episode from
delay_substeps_range). For the first d sub-steps of every env step the previous target is still applied,
then the new one. With decimation 4 and sim dt 5 ms, (0, 4) means 0-20 ms. (0, 0) = no delay.

The term keeps the name "JointPositionAction" in the env cfg because the task's startup event
randomize_joint_default_pos writes calibration noise into that term's _offset. That noise is kept here as
a bias on top of q_ref (offset - nominal offset), so training sees the same calibration error as the
original mimic task.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.string as string_utils
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ResidualJointPositionAction(JointPositionAction):
    cfg: ResidualJointPositionActionCfg

    def __init__(self, cfg: ResidualJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # offset before any startup randomization = the robot's nominal default pose
        self._nominal_offset = self._offset.clone()
        self._delta = torch.zeros_like(self._raw_actions)
        # per-joint residual bound
        self._limit = torch.full((self.action_dim,), float(cfg.residual_limit), device=self.device)
        if cfg.residual_limit_groups:
            idx, _, vals = string_utils.resolve_matching_names_values(cfg.residual_limit_groups, self._joint_names)
            self._limit[idx] = torch.tensor(vals, device=self.device)
        # actuation latency
        self._delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._substep = 0
        self._prev_target = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._vel_target = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._prev_vel_target = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._sample_delay(slice(None))

    @property
    def delta(self) -> torch.Tensor:
        """Residual actually added to q_ref in the last step [rad], after the clip."""
        return self._delta

    @property
    def residual_limit(self) -> torch.Tensor:
        """Per-joint bound delta_j [rad] (action order)."""
        return self._limit

    def _sample_delay(self, env_ids):
        lo, hi = self.cfg.delay_substeps_range
        n = self.num_envs if isinstance(env_ids, slice) else len(env_ids)
        self._delay[env_ids] = torch.randint(int(lo), int(hi) + 1, (n,), device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._fresh[ids] = True  # the robot was teleported: no "previous target" to hold
        self._sample_delay(ids)

    def process_actions(self, actions: torch.Tensor):
        # the target applied during the delay = last step's target
        if self._processed_actions is not None:
            self._prev_target[:] = self._processed_actions
            self._prev_vel_target[:] = self._vel_target
        self._raw_actions[:] = actions
        cmd = self._env.command_manager.get_term(self.cfg.command_name)
        k = cmd.time_steps + self.cfg.ref_lookahead
        if hasattr(cmd, "motion_end"):  # several clips laid end to end: never look into the next clip
            k = torch.minimum(k, cmd.motion_end - 1)
        else:
            k = torch.clamp(k, 0, cmd.motion.time_step_total - 1)
        q_ref = cmd.motion.joint_pos[k][:, self._joint_ids]  # npz joint order == sim order == action order
        if not self.cfg.ref_base:  # control experiment: plain mimic action
            self._delta = self._raw_actions * self._scale
            q_ref = self._nominal_offset
        elif self.cfg.ref_velocity_ff:
            self._vel_target[:] = cmd.motion.joint_vel[k][:, self._joint_ids]
        if self.cfg.ref_base:
            # residual_scale <= 0: per-joint scale from cfg.scale (the mimic task's 0.25 * effort / Kp)
            scale = self._scale if self.cfg.residual_scale <= 0 else self.cfg.residual_scale
            self._delta = torch.maximum(torch.minimum(self._raw_actions * scale, self._limit), -self._limit)
        calib = self._offset - self._nominal_offset  # zero unless add_joint_default_pos is active
        self._processed_actions = q_ref + calib + self._delta
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        self._prev_target[self._fresh] = self._processed_actions[self._fresh]
        self._prev_vel_target[self._fresh] = self._vel_target[self._fresh]
        self._fresh[:] = False
        self._substep = 0

    def apply_actions(self):
        if self.cfg.delay_substeps_range[1] > 0:
            hold = (self._substep < self._delay)[:, None]
            target = torch.where(hold, self._prev_target, self._processed_actions)
            vel_target = torch.where(hold, self._prev_vel_target, self._vel_target)
        else:
            target = self._processed_actions
            vel_target = self._vel_target
        self._substep += 1
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)
        if self.cfg.ref_velocity_ff:
            self._asset.set_joint_velocity_target(vel_target, joint_ids=self._joint_ids)


@configclass
class ResidualJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = ResidualJointPositionAction

    command_name: str = "motion"
    residual_scale: float = 0.25
    """policy output a -> a * residual_scale [rad] before the clip; <= 0 uses the per-joint cfg.scale"""
    residual_limit: float = 0.5
    """bound delta on every joint [rad] (joints not matched by residual_limit_groups)"""
    residual_limit_groups: dict[str, float] | None = None
    """per joint-name regex bound [rad], e.g. {".*_shoulder_.*": 0.17, ...}"""
    ref_lookahead: int = 1
    """base target = reference frame k + lookahead (1 = next frame, same as the Day 3 baseline)"""
    ref_base: bool = True
    """False = plain mimic action q_des = default + scale * a (control experiment, no residual)"""
    ref_velocity_ff: bool = False
    """also send the reference joint velocity as the PD velocity target (Kd acts on qd_ref - qd)"""
    delay_substeps_range: tuple[int, int] = (0, 0)
    """actuation latency in physics sub-steps, sampled per episode; (0, 0) = none"""
