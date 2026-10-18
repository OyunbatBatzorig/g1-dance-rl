"""Motion command that plays several reference clips (Day 6: one policy, several dances).

All clips are concatenated into one long timeline, so everything that reads the command
(observations, rewards, terminations, the residual action) works unchanged with a global frame index.
Each env remembers where its current clip ends (motion_end) and jumps to a newly sampled clip/frame
when it gets there, never running from one clip into the next.

Also used with a single clip (DR v2 tasks), for uniform_mix.

Sampling: first a clip (equal chance per clip by default, so a short dance is not drowned out by a long
one), then a 1-second bin inside that clip weighted by how often episodes failed there (the same
adaptive idea as the original MotionCommand, done per clip), then a uniform frame inside the bin.
"""

from __future__ import annotations

import math
import os
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from unitree_rl_lab.tasks.mimic.mdp.commands import MotionCommand, MotionCommandCfg, MotionLoader

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


class ConcatMotionLoader(MotionLoader):
    """Same attributes as MotionLoader, built from several npz files laid end to end."""

    def __init__(self, motion_files: Sequence[str], body_indexes, device: str = "cpu"):
        datas = []
        for f in motion_files:
            assert os.path.isfile(f), f"Invalid file path: {f}"
            datas.append(np.load(f))
        fps = [float(np.asarray(d["fps"]).reshape(-1)[0]) for d in datas]
        assert all(abs(x - fps[0]) < 1e-6 for x in fps), f"all clips must have the same fps, got {fps}"
        cat = {k: np.concatenate([d[k] for d in datas], axis=0) for k in _KEYS}
        self.fps = datas[0]["fps"]
        self.joint_pos = torch.tensor(cat["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(cat["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(cat["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(cat["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(cat["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(cat["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]
        self.clip_lengths = [d["joint_pos"].shape[0] for d in datas]
        self.clip_names = [os.path.splitext(os.path.basename(f))[0] for f in motion_files]


class MultiMotionCommand(MotionCommand):
    cfg: MultiMotionCommandCfg

    def __init__(self, cfg: MultiMotionCommandCfg, env: ManagerBasedRLEnv):
        cfg.motion_file = cfg.motion_files[0]  # the parent loads one file; replaced below
        super().__init__(cfg, env)
        self.motion = ConcatMotionLoader(cfg.motion_files, self.body_indexes, device=self.device)
        self.clip_names = self.motion.clip_names
        lengths = torch.tensor(self.motion.clip_lengths, dtype=torch.long, device=self.device)
        self.clip_start = torch.cumsum(lengths, 0) - lengths
        self.clip_end = self.clip_start + lengths  # exclusive
        n_clips = len(lengths)
        self.clip_weight = torch.tensor(
            cfg.clip_weights if cfg.clip_weights is not None else [1.0] * n_clips, device=self.device
        )
        self.clip_weight = self.clip_weight / self.clip_weight.sum()

        # ~1 s bins inside every clip
        frames_per_bin = int(round(1.0 / (env.cfg.decimation * env.cfg.sim.dt)))
        b_clip, b_start, b_len = [], [], []
        for c, (s, L) in enumerate(zip(self.clip_start.tolist(), self.motion.clip_lengths)):
            nb = max(1, math.ceil(L / frames_per_bin))
            edges = np.linspace(0, L, nb + 1).round().astype(int)
            for i in range(nb):
                b_clip.append(c)
                b_start.append(s + edges[i])
                b_len.append(max(1, edges[i + 1] - edges[i]))
        self.bin_clip = torch.tensor(b_clip, dtype=torch.long, device=self.device)
        self.bin_start = torch.tensor(b_start, dtype=torch.long, device=self.device)
        self.bin_len = torch.tensor(b_len, dtype=torch.long, device=self.device)
        self.bin_count = len(b_clip)
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)

        self.clip_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_end = self.clip_end[self.clip_id].clone()
        self.time_steps[:] = self.clip_start[self.clip_id]
        for name in self.clip_names:
            self.metrics[f"clip_share/{name}"] = torch.zeros(self.num_envs, device=self.device)
        print(f"[MultiMotionCommand] clips: {list(zip(self.clip_names, self.motion.clip_lengths))}, "
              f"{self.bin_count} bins, weights {self.clip_weight.tolist()}")

    def set_clip_start(self, env_ids, clip: int, frame: int = 0):
        """Put env_ids at a given clip/frame (used by the evaluation script)."""
        self.clip_id[env_ids] = clip
        self.motion_end[env_ids] = self.clip_end[clip]
        self.time_steps[env_ids] = self.clip_start[clip] + frame

    def _bin_of(self, t: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.searchsorted(self.bin_start, t, right=True) - 1, 0, self.bin_count - 1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            fail_bins = self._bin_of(self.time_steps[env_ids][episode_failed])
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # per clip: failure-weighted bins + a uniform floor, normalised inside the clip, times the clip weight
        p = torch.zeros(self.bin_count, device=self.device)
        for c in range(len(self.clip_names)):
            m = self.bin_clip == c
            pc = self.bin_failed_count[m] + self.cfg.adaptive_uniform_ratio / float(m.sum())
            p[m] = self.clip_weight[c] * pc / pc.sum()
        p = p / p.sum()
        if self.cfg.uniform_mix > 0:  # keep a share of starts spread over every bin, so one hard spot can't take over
            u = self.clip_weight[self.bin_clip] / torch.bincount(self.bin_clip)[self.bin_clip].float()
            p = (1 - self.cfg.uniform_mix) * p + self.cfg.uniform_mix * u / u.sum()

        n = len(env_ids)
        b = torch.multinomial(p, n, replacement=True)
        off = (sample_uniform(0.0, 1.0, (n,), device=self.device) * self.bin_len[b]).long()
        clip = self.bin_clip[b]
        self.clip_id[env_ids] = clip
        self.motion_end[env_ids] = self.clip_end[clip]
        self.time_steps[env_ids] = torch.minimum(self.bin_start[b] + off, self.clip_end[clip] - 1)

        H = -(p * (p + 1e-12).log()).sum()
        pmax, imax = p.max(dim=0)
        self.metrics["sampling_entropy"][:] = H / math.log(max(self.bin_count, 2))
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count
        for c, name in enumerate(self.clip_names):
            self.metrics[f"clip_share/{name}"][:] = (self.clip_id == c).float().mean()

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion_end)[0]
        self._resample_command(env_ids)
        self._update_relative_targets()

    def _update_relative_targets(self):
        # identical to the end of MotionCommand._update_command
        from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, yaw_quat

        nb = len(self.cfg.body_names)
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, nb, 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, nb, 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, nb, 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, nb, 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()


@configclass
class MultiMotionCommandCfg(MotionCommandCfg):
    class_type: type = MultiMotionCommand

    motion_files: list[str] = []
    clip_weights: list[float] | None = None
    """sampling weight per clip (default: equal)"""
    uniform_mix: float = 0.0
    """share of episode starts drawn uniformly over each clip's bins (rest: failure-weighted)"""
