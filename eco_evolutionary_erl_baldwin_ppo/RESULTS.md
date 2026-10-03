# Lifetime-private clipped-PPO investigation

## Method

Each organism has a private linear softmax actor, linear value baseline, and
on-policy rollout. The actor begins at inherited genome parameters; the critic
begins at zero. GAE advantages feed a clipped PPO objective with entropy
regularization, bounded actor/critic updates, and repeated epochs. Acquired
parameters and rollout state are discarded at reproduction. Policy sampling
and ecology use separate deterministic RNG streams.

## Development search

Seeds 63-65 screened actor learning rates 0.002 and 0.01, critic rates 0.01 and
0.05, discounts 0.5 and 0.9, rollout lengths 4 and 8, with four epochs. The
highest-mean candidate was seed-fragile. Post hoc, the frozen robust candidate was
`actor=0.01`, `critic=0.01`, `gamma=0.5`, rollout 4, epochs 4; it beat the
learning-off policy on all three development seeds by +26.7, +27.4, and +12.3
mean-trajectory agents. Because this robustness rule was not specified before
the screen, the later untouched and replication blocks—not the development
ranking—carry the evidential weight. Raw data: `development_search.csv`.

## Untouched gate

On seeds 66-73, enabled PPO averaged 111.5 agents versus 86.2 for its
policy-matched learning-off control and 100.6 for REINFORCE. Enabled minus off
was +25.3 with 7/8 wins, but its 95% paired t interval crossed zero (-3.3 to
+53.8; `p=0.074`). This was suggestive, not a passed gate.

## Independent replication

The predefined replication on seeds 74-81 did not reproduce the signal:
enabled PPO averaged 112.1 versus 110.0 learning-off. The difference was +2.1,
with 3/8 wins, a 95% interval of -28.5 to +32.6, and `p=0.878`. REINFORCE
averaged 124.5.

PPO-on versus PPO-off is the policy-stream-matched causal comparison. The
REINFORCE rows share integer seed labels but not common random-number streams,
because the baseline policy consumes the ecology RNG; comparisons to REINFORCE
are therefore descriptive rather than tightly paired controls.

## Decision

PPO learning is **not validated** as an alternative to the existing
REINFORCE-style lifetime update. The first gate's near-signal failed independent
replication, so no long-stability or multi-generation run is justified.

## Reproduction commands

```bash
python -m eco_evolutionary_erl_baldwin_ppo.calibrate_ppo --steps 300 --seeds 63,64,65 --alphas 0.002,0.01 --betas 0.01,0.05 --gammas 0.5,0.9 --rollouts 4,8 --epochs 4 --out eco_evolutionary_erl_baldwin_ppo/development_search.csv
python -m eco_evolutionary_erl_baldwin_ppo.calibrate_ppo --steps 300 --seeds 66,67,68,69,70,71,72,73 --alphas 0.01 --betas 0.01 --gammas 0.5 --rollouts 4 --epochs 4 --out eco_evolutionary_erl_baldwin_ppo/fresh_gate_300.csv
python -m eco_evolutionary_erl_baldwin_ppo.calibrate_ppo --steps 300 --seeds 74,75,76,77,78,79,80,81 --alphas 0.01 --betas 0.01 --gammas 0.5 --rollouts 4 --epochs 4 --out eco_evolutionary_erl_baldwin_ppo/replication_300.csv
```
