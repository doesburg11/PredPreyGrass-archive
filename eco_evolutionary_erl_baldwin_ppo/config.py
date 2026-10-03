"""Configuration for lifetime-private linear PPO."""

from copy import deepcopy
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.config import config_erl as _baseline

config_ppo = deepcopy(_baseline)
config_ppo.update({
    "ppo_actor_alpha": 0.01,
    "ppo_critic_beta": 0.01,
    "ppo_gamma": 0.5,
    "ppo_gae_lambda": 0.95,
    "ppo_clip_ratio": 0.2,
    "ppo_entropy_coef": 0.01,
    "ppo_rollout_length": 4,
    "ppo_epochs": 4,
    "ppo_actor_max_update_norm": 0.1,
    "ppo_critic_max_update_norm": 0.1,
    "ppo_terminal_bonus": 0.0,
})
config_erl = config_ppo
