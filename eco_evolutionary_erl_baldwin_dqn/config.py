"""Configuration for the lifetime linear DQN Baldwin variant."""

from copy import deepcopy
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.config import config_erl as _baseline

config_dqn = deepcopy(_baseline)
config_dqn.update({
    "dqn_alpha": 0.05,
    "dqn_gamma": 0.5,
    "dqn_epsilon": 0.75,
    "dqn_replay_capacity": 64,
    "dqn_batch_size": 8,
    "dqn_learning_starts": 8,
    "dqn_target_update_interval": 5,
    "dqn_huber_delta": 1.0,
    "dqn_max_update_norm": 0.1,
    "dqn_terminal_bonus": 0.0,
})
config_erl = config_dqn
