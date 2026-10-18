from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class ResidualPPORunnerCfg(BasePPORunnerCfg):
    # both Stand and full tasks log here, so the full run can --resume from a Stand run
    experiment_name = "g1_schuhplattler_residual"
    max_iterations = 3000
    save_interval = 250
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,  # residual starts near the reference (0.5 * 0.25 = ~0.12 rad exploration)
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )


@configclass
class MimicPPORunnerCfg(ResidualPPORunnerCfg):
    # control experiment with the original mimic action: Unitree's exploration noise
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
