import gymnasium as gym

_AGENT = f"{__name__}.rsl_rl_ppo_cfg:ResidualPPORunnerCfg"

gym.register(
    id="Unitree-G1-29dof-Mimic-Schuhplattler-Residual-Stand",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.residual_env_cfg:ResidualStandEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.residual_env_cfg:ResidualStandPlayEnvCfg",
        "rsl_rl_cfg_entry_point": _AGENT,
    },
)

gym.register(
    id="Unitree-G1-29dof-Mimic-Schuhplattler-Residual",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.residual_env_cfg:ResidualEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.residual_env_cfg:ResidualPlayEnvCfg",
        "rsl_rl_cfg_entry_point": _AGENT,
    },
)

gym.register(
    id="Unitree-G1-29dof-Mimic-Schuhplattler-Residual-Multi2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.residual_env_cfg:ResidualMulti2EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.residual_env_cfg:ResidualMulti2PlayEnvCfg",
        "rsl_rl_cfg_entry_point": _AGENT,
    },
)

gym.register(
    id="Unitree-G1-29dof-Mimic-Schuhplattler-Residual-Multi3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.residual_env_cfg:ResidualMulti3EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.residual_env_cfg:ResidualMulti3PlayEnvCfg",
        "rsl_rl_cfg_entry_point": _AGENT,
    },
)

# DR v2 (I-CTRL domain randomization)
# order B: Dance-102 first (pipeline check on a trusted reference), then + Gangnam, then + Schuhplattler
for _suffix, _cls in (
    ("DR-Stand", "DRStand"), ("DR", "DR"), ("DR-Multi2", "DRMulti2"), ("DR-Multi3", "DRMulti3"),
    ("DR-D102-Stand", "DRD102Stand"), ("DR-D102", "DRD102"), ("DR-D102-Multi2", "DRD102Multi2"),
    ("DR-D102-Multi3", "DRD102Multi3"),
):
    gym.register(
        id=f"Unitree-G1-29dof-Mimic-Schuhplattler-Residual-{_suffix}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.residual_env_cfg:{_cls}EnvCfg",
            "play_env_cfg_entry_point": f"{__name__}.residual_env_cfg:{_cls}PlayEnvCfg",
            "rsl_rl_cfg_entry_point": _AGENT,
        },
    )

# control experiments (Dance-102, from scratch)
for _id, _cls, _agent in (
    ("Unitree-G1-29dof-Mimic-Schuhplattler-Residual-D102-NoDR", "ResD102NoDR", _AGENT),
    ("Unitree-G1-29dof-Mimic-Schuhplattler-MimicAction-D102-DR", "MimicD102DR",
     f"{__name__}.rsl_rl_ppo_cfg:MimicPPORunnerCfg"),
    ("Unitree-G1-29dof-Mimic-Schuhplattler-Residual-DR-D102-Wide", "DRD102Wide",
     f"{__name__}.rsl_rl_ppo_cfg:MimicPPORunnerCfg"),
    # Policy 2 / 3 (Wide residual): Dance-102 + Gangnam (trimmed), + Schuhplattler (calm ending, grounded)
    ("Unitree-G1-29dof-Mimic-Schuhplattler-Residual-DR-D102-Wide-Multi2", "DRD102WideMulti2",
     f"{__name__}.rsl_rl_ppo_cfg:MimicPPORunnerCfg"),
    ("Unitree-G1-29dof-Mimic-Schuhplattler-Residual-DR-D102-Wide-Multi3", "DRD102WideMulti3",
     f"{__name__}.rsl_rl_ppo_cfg:MimicPPORunnerCfg"),
    # Schuhplattler alone: residual (Wide) vs full policy (MimicAction), same DR v2 and runner
    ("Unitree-G1-29dof-Mimic-Schuhplattler-Residual-DR-Sch-Wide", "DRSchWide",
     f"{__name__}.rsl_rl_ppo_cfg:MimicPPORunnerCfg"),
    ("Unitree-G1-29dof-Mimic-Schuhplattler-MimicAction-DR-Sch", "MimicSchDR",
     f"{__name__}.rsl_rl_ppo_cfg:MimicPPORunnerCfg"),
    # Schuhplattler alone, balance-fixed reference (Wide residual)
    ("Unitree-G1-29dof-Mimic-Schuhplattler-Residual-DR-SchBal-Wide", "DRSchBalWide",
     f"{__name__}.rsl_rl_ppo_cfg:MimicPPORunnerCfg"),
    ("Unitree-G1-29dof-Mimic-Schuhplattler-MimicAction-DR-SchBal", "MimicSchBalDR",
     f"{__name__}.rsl_rl_ppo_cfg:MimicPPORunnerCfg"),
    # Policy 3 (Wide residual) with the improved balance-fixed Schuhplattler (schuhplattler_gmr_end_bal2.npz)
    ("Unitree-G1-29dof-Mimic-Schuhplattler-Residual-DR-D102-Wide-Multi3Bal", "DRD102WideMulti3Bal",
     f"{__name__}.rsl_rl_ppo_cfg:MimicPPORunnerCfg"),
    ("Unitree-G1-29dof-Mimic-Schuhplattler-MimicAction-DR-D102-Multi3Bal", "MimicD102Multi3BalDR",
     f"{__name__}.rsl_rl_ppo_cfg:MimicPPORunnerCfg"),
    ("Unitree-G1-29dof-Mimic-Schuhplattler-MimicAction-DR-SchBal2", "MimicSchBal2DR",
     f"{__name__}.rsl_rl_ppo_cfg:MimicPPORunnerCfg"),
):
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.residual_env_cfg:{_cls}EnvCfg",
            "play_env_cfg_entry_point": f"{__name__}.residual_env_cfg:{_cls}PlayEnvCfg",
            "rsl_rl_cfg_entry_point": _agent,
        },
    )
