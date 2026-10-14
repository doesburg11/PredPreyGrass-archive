"""Local Ray output location for this archived experiment."""
from pathlib import Path

RAY_RESULTS_DIR = Path(__file__).resolve().parent / "runs"

POLICY_IDS = ('prey', 'predator_male', 'predator_female')

def _has_policy_modules(checkpoint: Path) -> bool:
    modules = checkpoint / "learner_group" / "learner" / "rl_module"
    for policy in POLICY_IDS:
        if not all((modules / policy / name).is_file() for name in
                   ("metadata.json", "class_and_ctor_args.pkl", "module_state.pkl")):
            return False
    return True


def latest_checkpoint() -> Path:
    """Find the newest standard-treatment run with an RLlib module checkpoint."""
    experiments = sorted(
        (p for p in RAY_RESULTS_DIR.iterdir()
         if p.is_dir() and not any(tag in p.name for tag in ("_NEUTRAL_CONTROL_", "_SCALED_", "_FIXED"))),
        key=lambda p: (p.name[-19:], p.name), reverse=True,
    )
    for experiment in experiments:
        candidates = [
            p.parent for p in experiment.glob("*/checkpoint*/rllib_checkpoint.json")
            if _has_policy_modules(p.parent)
        ]
        candidates.extend(
            p.parent for p in experiment.glob("checkpoint*/rllib_checkpoint.json")
            if _has_policy_modules(p.parent)
        )
        if candidates:
            def checkpoint_order(path):
                suffix = path.name.rsplit("_", 1)[-1]
                return (int(suffix) if suffix.isdigit() else -1, path.stat().st_mtime_ns, str(path))
            return max(candidates, key=checkpoint_order)
    raise FileNotFoundError(f"No RLlib module checkpoint found under {RAY_RESULTS_DIR}")
