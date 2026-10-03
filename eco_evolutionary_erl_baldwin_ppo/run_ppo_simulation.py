"""Run lifetime PPO through the baseline simulation CLI."""
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin import run_erl_simulation as runner
from .config import config_ppo
from .world import PpoWorld

def main():
    runner.config_erl=config_ppo;runner.ErlWorld=PpoWorld;runner.EXPECTED_WORLD_CLASS=PpoWorld;runner.RUN_NAME_PREFIX="erl_baldwin_ppo";runner.main()
if __name__=="__main__":main()
