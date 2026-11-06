"""Day 4 residual-RL tasks on top of the Schuhplattler mimic task.

Everything (scene, observations, rewards, terminations, randomization) is inherited unchanged from
../tracking_env_cfg.py. Only three things differ:
  * the action is the bounded residual on q_ref (residual_action.py),
  * the motion is the Day 2 GMR retarget in ~/projects/g1_pilot (not the npz next to the original task),
  * one small extra penalty keeps the residual small (residual_l2).
"""

import dataclasses
import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import unitree_rl_lab.tasks.mimic.mdp as mdp
from unitree_rl_lab.tasks.mimic.robots.g1_29dof.schuhplattler.tracking_env_cfg import RewardsCfg, RobotEnvCfg

from .multi_motion import MultiMotionCommand, MultiMotionCommandCfg
from .residual_action import ResidualJointPositionActionCfg

# motion files live in a workspace folder (day2/, day4/, day6/); override with G1_DANCE_DATA
DATA_DIR = os.path.expanduser(os.environ.get("G1_DANCE_DATA", "~/projects/g1_pilot"))
FULL_MOTION = os.path.join(DATA_DIR, "day2/schuhplattler_gmr.npz")
STAND_MOTION = os.path.join(DATA_DIR, "day4/schuhplattler_stand.npz")
DANCE102_MOTION = os.path.join(DATA_DIR, "day6/dance_102.npz")
GANGNAM_MOTION = os.path.join(DATA_DIR, "day6/gangnam_style.npz")


@configclass
class ResidualActionsCfg:
    JointPositionAction = ResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,  # unused by the residual term, kept for deploy.yaml export
        use_default_offset=True,
        residual_scale=0.25,
        residual_limit=0.5,
        ref_lookahead=1,
    )


@configclass
class ResidualRewardsCfg(RewardsCfg):
    residual_l2 = RewTerm(func=mdp.action_l2, weight=-0.01)


@configclass
class ResidualEnvCfg(RobotEnvCfg):
    actions: ResidualActionsCfg = ResidualActionsCfg()
    rewards: ResidualRewardsCfg = ResidualRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = FULL_MOTION


class ResidualPlayEnvCfg(ResidualEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9


class ResidualStandEnvCfg(ResidualEnvCfg):
    """Stage A: only the first 1.8 s (stand still, then raise both arms overhead)."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = STAND_MOTION


class ResidualStandPlayEnvCfg(ResidualStandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9


def _multi_motion_cfg(single, motion_files):
    """Copy every setting of the single-clip motion command, then point it at several clips."""
    multi = MultiMotionCommandCfg()
    for f in dataclasses.fields(single):
        setattr(multi, f.name, getattr(single, f.name))
    multi.class_type = MultiMotionCommand
    multi.motion_files = list(motion_files)
    multi.motion_file = motion_files[0]
    return multi


class ResidualMulti2EnvCfg(ResidualEnvCfg):
    """Day 6, policy 2: Schuhplattler + Dance-102, equal chance per clip."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion = _multi_motion_cfg(self.commands.motion, [FULL_MOTION, DANCE102_MOTION])


class ResidualMulti2PlayEnvCfg(ResidualMulti2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9


class ResidualMulti3EnvCfg(ResidualEnvCfg):
    """Day 6, policy 3: Schuhplattler + Dance-102 + Gangnam Style, equal chance per clip."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion = _multi_motion_cfg(
            self.commands.motion, [FULL_MOTION, DANCE102_MOTION, GANGNAM_MOTION]
        )


class ResidualMulti3PlayEnvCfg(ResidualMulti3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9


# ---------------------------------------------------------------------------------------------------
# DR v2: I-CTRL (arXiv 2405.08726, Table II) domain randomization on top of the tasks above.
# Sensor noise, motor PD gains, link masses, friction, per-group residual bound, actuation latency.
# ---------------------------------------------------------------------------------------------------

from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise  # noqa: E402

# I-CTRL: arms 10 deg, legs and torso 30 deg. Arms raised to 20 deg (2026-10-14): with 0.17 Dance-102 needed
# more than the bound in 71% of frames just to cancel the soft PD's lag.
DR_RESIDUAL_LIMITS = {
    ".*_shoulder_.*": 0.35,
    ".*_elbow_.*": 0.35,
    ".*_wrist_.*": 0.35,
    ".*_hip_.*": 0.52,
    ".*_knee_.*": 0.52,
    ".*_ankle_.*": 0.52,
    "waist_.*": 0.52,
}


# grounded references (tools/npz_ground.py): the support foot always touches the floor
FULL_GROUNDED = os.path.join(DATA_DIR, "day2/schuhplattler_gmr_grounded.npz")
STAND_GROUNDED = os.path.join(DATA_DIR, "day4/schuhplattler_stand_grounded.npz")
DR_UNIFORM_MIX = 0.0  # 0.5 halved Stand episode length (162 vs 335, 2026-10-14); 0 = Unitree sampler per clip


def apply_dr_v2(cfg, motion_files, randomize=True):
    """Switch a residual env cfg to DR v2 (in place), playing motion_files.

    randomize=False keeps the v2 action (group bounds, velocity feed-forward) but none of the DR v2
    randomization (control experiment: original sensor noise, gains, masses, friction, no latency).
    """
    # -- motion: grounded clip(s), sampler that can't collapse onto one hard second
    cfg.commands.motion = _multi_motion_cfg(cfg.commands.motion, motion_files)
    cfg.commands.motion.uniform_mix = DR_UNIFORM_MIX
    # -- action: per-group bound
    cfg.actions.JointPositionAction.residual_limit_groups = dict(DR_RESIDUAL_LIMITS)
    # -- PD velocity feed-forward: Kd acts on (qd_ref - qd), removes the ~0.064 s * qd lag of the soft G1 PD
    cfg.actions.JointPositionAction.ref_velocity_ff = True
    if not randomize:
        return
    # -- sensors (I-CTRL: ang vel 0.1, joint pos 0.1, joint vel 1.0; ours keeps the stronger 0.2 ang vel and
    #    starts joint pos at 0.05, see the plan's open risks)
    cfg.observations.policy.base_ang_vel.noise = Unoise(n_min=-0.2, n_max=0.2)
    cfg.observations.policy.joint_pos_rel.noise = Unoise(n_min=-0.05, n_max=0.05)
    cfg.observations.policy.joint_vel_rel.noise = Unoise(n_min=-1.0, n_max=1.0)
    # -- motor: Kp and Kd scaled x0.75-1.25, fixed per robot
    cfg.events.actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.25),
            "damping_distribution_params": (0.75, 1.25),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    # -- dynamics: every link mass x0.7-1.3 (torso CoM shift from the original task stays on)
    cfg.events.link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    # -- environment: wider friction (I-CTRL x0.5-2.0 around 1.0)
    cfg.events.physics_material.params["static_friction_range"] = (0.3, 2.0)
    cfg.events.physics_material.params["dynamic_friction_range"] = (0.3, 1.6)
    # -- action: 0-20 ms latency (4 sub-steps of 5 ms)
    cfg.actions.JointPositionAction.delay_substeps_range = (0, 4)


def use_mimic_action(cfg):
    """Control experiment: the original Unitree / BeyondMimic action (default + scale * a, no residual)."""
    from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_ACTION_SCALE

    act = cfg.actions.JointPositionAction
    act.ref_base = False
    act.ref_velocity_ff = False
    act.scale = dict(UNITREE_G1_29DOF_MIMIC_ACTION_SCALE)
    cfg.rewards.residual_l2 = None  # the original task has no such penalty


class DRStandEnvCfg(ResidualStandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, [STAND_GROUNDED])


class DRStandPlayEnvCfg(ResidualStandPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, [STAND_GROUNDED])


class DREnvCfg(ResidualEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, [FULL_GROUNDED])


class DRPlayEnvCfg(ResidualPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, [FULL_GROUNDED])


class DRMulti2EnvCfg(ResidualMulti2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, [FULL_GROUNDED, DANCE102_MOTION])


class DRMulti2PlayEnvCfg(ResidualMulti2PlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, [FULL_GROUNDED, DANCE102_MOTION])


class DRMulti3EnvCfg(ResidualMulti3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, [FULL_GROUNDED, DANCE102_MOTION, GANGNAM_MOTION])


class DRMulti3PlayEnvCfg(ResidualMulti3PlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, [FULL_GROUNDED, DANCE102_MOTION, GANGNAM_MOTION])


# DR v2, order B (2026-10-14): start from a trusted reference to check the pipeline first.
# Policy 1 = Dance-102 (Unitree mocap), 2 = + Gangnam Style, 3 = + Schuhplattler (grounded).
D102_STAND = os.path.join(DATA_DIR, "day6/dance_102_stand.npz")  # day4/make_stand_npz.py
D102_ORDER = {
    "D102Stand": [D102_STAND],
    "D102": [DANCE102_MOTION],
    "D102Multi2": [DANCE102_MOTION, GANGNAM_MOTION],
    "D102Multi3": [DANCE102_MOTION, GANGNAM_MOTION, FULL_GROUNDED],
}



class DRD102StandEnvCfg(ResidualEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, D102_ORDER["D102Stand"])


class DRD102StandPlayEnvCfg(ResidualPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, D102_ORDER["D102Stand"])


class DRD102EnvCfg(ResidualEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, D102_ORDER["D102"])


class DRD102PlayEnvCfg(ResidualPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, D102_ORDER["D102"])


class DRD102Multi2EnvCfg(ResidualEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, D102_ORDER["D102Multi2"])


class DRD102Multi2PlayEnvCfg(ResidualPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, D102_ORDER["D102Multi2"])


class DRD102Multi3EnvCfg(ResidualEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, D102_ORDER["D102Multi3"])


class DRD102Multi3PlayEnvCfg(ResidualPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, D102_ORDER["D102Multi3"])


# Control experiments on Dance-102 (2026-10-14): from scratch, 300 iterations, to find what slows learning.
# Residual-D102-NoDR: residual action as in DR-D102, without the DR v2 randomization.
# Mimic-D102-DR: Unitree's action (no residual) with the full DR v2 randomization incl. latency.
class ResD102NoDREnvCfg(ResidualEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, [DANCE102_MOTION], randomize=False)


class ResD102NoDRPlayEnvCfg(ResidualPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, [DANCE102_MOTION], randomize=False)


class MimicD102DREnvCfg(ResidualEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, [DANCE102_MOTION])
        use_mimic_action(self)


class MimicD102DRPlayEnvCfg(ResidualPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, [DANCE102_MOTION])
        use_mimic_action(self)


# Residual with the mimic task's exploration (2026-10-14): same per-joint action scale as Unitree
# (0.25 * effort / Kp), bound 1.0 rad on every joint, no residual_l2 penalty, init noise 1.0 (MimicPPORunnerCfg).
# The reference stays the base of the action (q_ref + delta), DR v2 on.
WIDE_RESIDUAL_LIMIT = 1.0


def use_wide_residual(cfg):
    from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_ACTION_SCALE

    act = cfg.actions.JointPositionAction
    act.residual_scale = 0.0
    act.scale = dict(UNITREE_G1_29DOF_MIMIC_ACTION_SCALE)
    act.residual_limit_groups = None
    act.residual_limit = WIDE_RESIDUAL_LIMIT
    cfg.rewards.residual_l2 = None


class DRD102WideEnvCfg(ResidualEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, [DANCE102_MOTION])
        use_wide_residual(self)


class DRD102WidePlayEnvCfg(ResidualPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, [DANCE102_MOTION])
        use_wide_residual(self)


# Policy 2 / 3 on the Wide residual (2026-10-16), each trained from scratch like Policy 1 (DR-D102-Wide).
# Gangnam without its first 3 frames (27 rad/s velocity spike at the start of the Unitree CSV);
# Schuhplattler v2 with the calm ending (11.23 s cut, 1.6 s blend to the default pose, 1.5 s hold), grounded.
GANGNAM_TRIM = os.path.join(DATA_DIR, "day6/gangnam_style_trim.npz")
SCH_END_GROUNDED = os.path.join(DATA_DIR, "day2/schuhplattler_gmr_end_grounded.npz")
WIDE_ORDER = {
    "Multi2": [DANCE102_MOTION, GANGNAM_TRIM],
    "Multi3": [DANCE102_MOTION, GANGNAM_TRIM, SCH_END_GROUNDED],
}


class DRD102WideMulti2EnvCfg(ResidualEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, WIDE_ORDER["Multi2"])
        use_wide_residual(self)


class DRD102WideMulti2PlayEnvCfg(ResidualPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, WIDE_ORDER["Multi2"])
        use_wide_residual(self)


class DRD102WideMulti3EnvCfg(ResidualEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, WIDE_ORDER["Multi3"])
        use_wide_residual(self)


class DRD102WideMulti3PlayEnvCfg(ResidualPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, WIDE_ORDER["Multi3"])
        use_wide_residual(self)


# Schuhplattler alone (2026-10-17): residual vs full policy on the same reference, DR v2 and runner,
# from scratch. The only difference is the action base: q_ref + delta (Wide) vs default + scale * a (Mimic).
class DRSchWideEnvCfg(ResidualEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, [SCH_END_GROUNDED])
        use_wide_residual(self)


class DRSchWidePlayEnvCfg(ResidualPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, [SCH_END_GROUNDED])
        use_wide_residual(self)


class MimicSchDREnvCfg(ResidualEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, [SCH_END_GROUNDED])
        use_mimic_action(self)


class MimicSchDRPlayEnvCfg(ResidualPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, [SCH_END_GROUNDED])
        use_mimic_action(self)


# Schuhplattler alone on the balance-fixed reference (2026-10-17, tools/balance_fix.py: left foot pinned 0.7-1.35 s,
# pelvis over the stance foot in single support). Same as DRSchWide otherwise, so it compares with sch_res directly.
SCH_BAL = os.path.join(DATA_DIR, "day2/schuhplattler_gmr_end_bal.npz")


class DRSchBalWideEnvCfg(ResidualEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, [SCH_BAL])
        use_wide_residual(self)


class DRSchBalWidePlayEnvCfg(ResidualPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, [SCH_BAL])
        use_wide_residual(self)


class MimicSchBalDREnvCfg(ResidualEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, [SCH_BAL])
        use_mimic_action(self)


class MimicSchBalDRPlayEnvCfg(ResidualPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, [SCH_BAL])
        use_mimic_action(self)


# Policy 3 on the improved balance-fixed Schuhplattler (2026-10-17, v2 of balance_fix): no hard pin (the 0.7-1.35 s
# pin teleported the left foot 22 cm), orientation IK working, the 10.35-10.55 s hop replaced by double support,
# right foot put down quickly at the end. Otherwise identical to DRD102WideMulti3, so it compares with d102_p3.
SCH_BAL2 = os.path.join(DATA_DIR, "day2/schuhplattler_gmr_end_bal2.npz")
WIDE_ORDER["Multi3Bal"] = [DANCE102_MOTION, GANGNAM_TRIM, SCH_BAL2]


class DRD102WideMulti3BalEnvCfg(ResidualEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, WIDE_ORDER["Multi3Bal"])
        use_wide_residual(self)


class DRD102WideMulti3BalPlayEnvCfg(ResidualPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, WIDE_ORDER["Multi3Bal"])
        use_wide_residual(self)


# Same three dances as DRD102WideMulti3Bal, full policy (MimicAction) instead of the Wide residual:
# on Schuhplattler alone the full policy was far ahead in MuJoCo (sch_full 90.6% vs sch_res 0%).
class MimicD102Multi3BalDREnvCfg(ResidualEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, WIDE_ORDER["Multi3Bal"])
        use_mimic_action(self)


class MimicD102Multi3BalDRPlayEnvCfg(ResidualPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, WIDE_ORDER["Multi3Bal"])
        use_mimic_action(self)


# Schuhplattler alone on the v2 reference, full policy: compares with sch_full (same action, DR v2, 3000 it; old ref).
class MimicSchBal2DREnvCfg(ResidualEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, [SCH_BAL2])
        use_mimic_action(self)


class MimicSchBal2DRPlayEnvCfg(ResidualPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, [SCH_BAL2])
        use_mimic_action(self)


# -- v3s reference (Oyunbat's own edits + smooth tempo, 15.0 s) --
SCH_V3S = os.path.join(DATA_DIR, "day2/schuhplattler_gmr_v3s.npz")
FLOOR_BUMP_M = 0.01  # DR+ floor height noise (m)


def apply_dr_plus(cfg):
    """DR v2 plus sim2real extras for the real G1 (call after apply_dr_v2).

    Latency stays 0-20 ms: residual_action holds the old target for at most one env step (4 sub-steps).
    """
    # joint zero / encoder calibration offset (original task: +-0.01 rad)
    cfg.events.add_joint_default_pos.params["pos_distribution_params"] = (-0.03, 0.03)
    # torso CoM shift: battery, cables, payload (original: x +-0.025)
    cfg.events.base_com.params["com_range"] = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)}
    # motor rotor inertia (armature) x0.8-1.2, fixed per robot
    cfg.events.joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    # IMU orientation drift (original: +-0.05)
    cfg.observations.policy.motion_anchor_ori_b.noise = Unoise(n_min=-0.08, n_max=0.08)
    # floor: friction is already 0.3-2.0 (apply_dr_v2); add a slightly uneven floor (0-1 cm bumps, 10 cm grid)
    from isaaclab.terrains import TerrainGeneratorCfg
    from isaaclab.terrains.height_field import HfRandomUniformTerrainCfg

    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=10,
        num_cols=20,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "rough": HfRandomUniformTerrainCfg(
                proportion=1.0, noise_range=(0.0, FLOOR_BUMP_M), noise_step=0.005, border_width=0.25
            )
        },
    )
    cfg.scene.terrain.max_init_terrain_level = None


# Schuhplattler v3s alone, full policy, DR v2: same setup as sch_bal2_full (v2 ref) -> shows what the new ref changes.
class MimicSchV3sDREnvCfg(ResidualEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, [SCH_V3S])
        use_mimic_action(self)


class MimicSchV3sDRPlayEnvCfg(ResidualPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, [SCH_V3S])
        use_mimic_action(self)


# Same, with the sim2real extras (apply_dr_plus): the candidate for MuJoCo -> real G1.
class MimicSchV3sDRPlusEnvCfg(ResidualEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, [SCH_V3S])
        apply_dr_plus(self)
        use_mimic_action(self)


class MimicSchV3sDRPlusPlayEnvCfg(ResidualPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, [SCH_V3S])
        apply_dr_plus(self)
        use_mimic_action(self)


# -- floor mix for sim2real: wood / smooth / stone / slightly sloped floors (call after apply_dr_plus) --
def apply_floor_mix(cfg):
    """Each robot gets one floor tile: flat, rough 0-1 cm, stone-like 0-2.5 cm, or a gentle slope (<= ~3 deg).

    Floor type = geometry here; slipperiness (smooth tile vs wood vs stone) comes from the friction range below.
    """
    from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainGeneratorCfg
    from isaaclab.terrains.height_field import HfPyramidSlopedTerrainCfg, HfRandomUniformTerrainCfg

    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=10,
        num_cols=20,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "flat": MeshPlaneTerrainCfg(proportion=0.3),
            "rough": HfRandomUniformTerrainCfg(
                proportion=0.3, noise_range=(0.0, 0.01), noise_step=0.005, border_width=0.25
            ),
            "stone": HfRandomUniformTerrainCfg(
                proportion=0.2, noise_range=(0.0, 0.025), noise_step=0.005, border_width=0.25
            ),
            "slope": HfPyramidSlopedTerrainCfg(
                proportion=0.2, slope_range=(0.0, 0.05), platform_width=0.5, border_width=0.25
            ),
        },
    )
    cfg.scene.terrain.max_init_terrain_level = None
    # slippery polished floor (0.2) up to grippy rubber mat (2.0); combine mode is multiply with ground 1.0
    cfg.events.physics_material.params["static_friction_range"] = (0.2, 2.0)
    cfg.events.physics_material.params["dynamic_friction_range"] = (0.15, 1.6)


# Same as DRPlus-SchV3s on the floor mix: fine-tune from sch_v3s_plus (same motion, same action).
class MimicSchV3sDRFloorEnvCfg(ResidualEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, [SCH_V3S])
        apply_dr_plus(self)
        apply_floor_mix(self)
        use_mimic_action(self)


class MimicSchV3sDRFloorPlayEnvCfg(ResidualPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_dr_v2(self, [SCH_V3S])
        apply_dr_plus(self)
        apply_floor_mix(self)
        use_mimic_action(self)
