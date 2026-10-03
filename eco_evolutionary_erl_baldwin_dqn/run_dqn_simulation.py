"""Run lifetime DQN using the baseline simulation CLI."""
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin import run_erl_simulation as runner
from .config import config_dqn
from .world import DqnWorld

def main():
    runner.config_erl=config_dqn; runner.ErlWorld=DqnWorld
    runner.EXPECTED_WORLD_CLASS=DqnWorld; runner.RUN_NAME_PREFIX="ERL_BALDWIN_DQN"; runner.main()

if __name__ == "__main__": main()
