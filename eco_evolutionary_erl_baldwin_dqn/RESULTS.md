# Lifetime Double-DQN investigation

## Mechanism

Each organism owns a linear live Q function, target Q function, 64-transition
replay buffer, and update counter. Newborns initialize both Q functions from
their genome and start with empty replay; none of the acquired state is
inherited. Updates use Double-DQN targets, batches of eight, Huber-clipped TD
errors, and a maximum parameter-step norm of 0.1. Fatal actions add one terminal
transition with zero bootstrap.

## Development and exploration rescue

The initial low-exploration grid (`epsilon` 0.05 or 0.1) was numerically stable
but behaviorally collapsed: random inherited Q values produced nearly
deterministic founder policies. A labeled rescue screen tested broader
exploration. `alpha=0.02`, `epsilon=0.75`, and target synchronization every 10
updates averaged 194.2 agents over the common 300-step horizon on seeds 41-43,
versus 133.5 for REINFORCE. Raw screens are `calibration.csv` and
`exploration_rescue.csv`.

## Held-out short-horizon result

The frozen setting survived all fresh seeds 44-51 and beat REINFORCE on every
matched initialization seed.

| Treatment | Mean trajectory population |
|---|---:|
| REINFORCE | 126.3 |
| DQN, learning off, epsilon 0.75 | 174.9 |
| DQN, learning enabled | 189.7 |

DQN minus REINFORCE was +63.4 agents (95% paired t interval +35.1 to +91.7;
`p=0.0011`). However, learning-enabled DQN minus its policy-matched learning-off
twin was only +14.7, with a wide interval crossing zero (-27.4 to +56.8;
`p=0.435`). Raw results including the required ablation are in
`heldout_with_off_300.csv`.

These original runs predated random-stream isolation: replay sampling consumed
the ecology RNG, so they matched initial conditions but not step-for-step random
draws. The follow-up below removes that confound and supersedes this comparison.

## Long stability result

Before the learning-off confound was measured, the frozen learning-enabled DQN
was run for 3,000 steps on seeds 52-54. It survived 3/3, as did REINFORCE, with
mean common-horizon populations of 287.2 and 162.0 respectively. Sampled Q
maxima stayed below 2.16 and no numerical failure or 2,000-agent cap event was
observed. Raw results are in `stability_3000.csv`.

## Decision

The implementation is stable and the overall DQN condition is ecologically
strong, but the present experiment does not show that DQN *learning* caused the
gain. Most of the advantage is already present when learning is disabled and
75% of actions are random. Therefore this is not yet a validated alternative
learner for the Baldwin experiment. It must not advance to a multi-generation
claim until learning-enabled DQN reproducibly beats the exploration-matched
learning-off control.

## Exploration-matched follow-up with isolated random streams

The follow-up separated ecology, epsilon exploration, and replay sampling into
independent deterministic RNG streams. Replay minibatch selection can therefore
no longer perturb subsequent ecological or exploratory random draws.

At fixed `epsilon=0.75`, a development screen on seeds 41-43 crossed learning
rates 0.005, 0.02, and 0.05; discount factors 0.5, 0.9, and 1.0; and target-sync
intervals 5, 10, and 50. The selected setting (`alpha=0.05`, `gamma=0.5`, target
sync every 5 updates) beat learning-off on all three development seeds, by a
mean 72.4 agents. Raw results are in `exploration_matched_search.csv`.

That apparent learning benefit did **not** reproduce on confirmation seeds
44-51. These seeds had been inspected in the earlier pre-isolation experiment,
so this is a reanalysis/confirmation set rather than a formal untouched gate:

| Treatment | Mean trajectory population |
|---|---:|
| REINFORCE | 126.3 |
| DQN, learning off, epsilon 0.75 | 176.8 |
| Selected DQN, learning enabled | 177.5 |

Learning-enabled minus learning-off was +0.8 agents, won only 4/8 seed pairs,
and had a 95% paired t interval of -25.4 to +27.0 (`p=0.946`). Enabled DQN still
beat REINFORCE by +51.3 agents (`p=0.0107`), but that comparison remains dominated
by the exploration policy. Raw results are in
`exploration_matched_heldout_300.csv`.

The formal gate then used genuinely untouched seeds 55-62. Enabled DQN averaged
182.6 agents versus 155.1 for learning-off. The paired difference was +27.5,
with 5/8 wins and a 95% paired t interval of -18.4 to +73.4 (`p=0.200`). It beat
REINFORCE (130.2) on all 8 seeds, but again did not reliably beat its
exploration-matched control. Raw results are in
`exploration_matched_fresh_gate_300.csv`.

**Updated decision:** the exploration-matched search failed both confirmation
and formal untouched gates. Replay plus a target network is therefore not a
validated alternative lifetime learner in this environment, and no
multi-generation or long-stability run is justified for this candidate.

## Reproduction commands

```bash
python -m eco_evolutionary_erl_baldwin_dqn.calibrate_dqn --steps 300 --seeds 41,42,43 --alphas 0.005,0.02,0.05 --epsilons 0.75 --gammas 0.5,0.9,1.0 --targets 5,10,50 --out eco_evolutionary_erl_baldwin_dqn/exploration_matched_search.csv
python -m eco_evolutionary_erl_baldwin_dqn.calibrate_dqn --steps 300 --seeds 44,45,46,47,48,49,50,51 --alphas 0.05 --epsilons 0.75 --gammas 0.5 --targets 5 --out eco_evolutionary_erl_baldwin_dqn/exploration_matched_heldout_300.csv
python -m eco_evolutionary_erl_baldwin_dqn.calibrate_dqn --steps 300 --seeds 55,56,57,58,59,60,61,62 --alphas 0.05 --epsilons 0.75 --gammas 0.5 --targets 5 --out eco_evolutionary_erl_baldwin_dqn/exploration_matched_fresh_gate_300.csv
```
