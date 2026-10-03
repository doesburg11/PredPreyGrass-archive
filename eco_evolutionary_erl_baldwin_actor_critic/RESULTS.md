# Linear actor-critic investigation

## Design

The inherited action matrix initializes a private lifetime actor. Each newborn
receives a zero-initialized linear state-value critic. Actor changes and critic
state are never inherited. The critic uses one-step TD(0); the actor uses the
existing softmax score gradient modulated by the TD error. Fatal actions receive
one terminal update with zero bootstrap. Actor and critic update norms are capped
at 0.1 and 0.5 respectively.

## Development calibration

Seeds 41-43 ran for 300 steps. The grid was actor alpha `{0.002, 0.01, 0.05}`,
critic beta `{2, 5} * alpha`, and gamma `{0, 0.9}`. All 12 settings were finite
and avoided extinction. The selected setting was `alpha=0.002`, `beta=0.01`,
`gamma=0.9`: its mean common-horizon population was 134.9, versus 133.5 for
REINFORCE. Selection used the highest three-seed mean after excluding failures;
with 12 candidates and only three development seeds, the 1.4-agent advantage
is a selection result, not evidence of superiority. Raw results are in
`calibration.csv`. Diagnostic fields ending in `_max_seen` are periodic
living-agent samples every 50 steps, not exact maxima over every lifetime.

## Held-out result

The frozen setting was evaluated without retuning on fresh seeds 44-51 for 300
steps. Neither treatment went extinct and there were no numerical failures.

| Treatment | Mean trajectory population | Mean end population |
|---|---:|---:|
| REINFORCE | 126.3 | 173.3 |
| Actor-critic | 103.6 | 117.4 |

Actor-critic underperformed REINFORCE on 6/8 matched seeds. The paired
actor-critic-minus-REINFORCE trajectory difference was -22.7 agents (95% t
interval -44.2 to -1.2; paired t-test `p=0.041`). Raw results are in
`heldout_300.csv`.

The held-out gate failed, so the planned 3,000-step stability test and expensive
multi-generation study were not launched. This formulation is a tested negative
result, not a validated replacement for REINFORCE.
