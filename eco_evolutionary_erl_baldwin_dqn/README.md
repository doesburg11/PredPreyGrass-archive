# ERL Baldwin with lifetime-private linear Double DQN

This variant tests a genuinely value-based alternative with the main stability
features absent from the failed online SARSA experiment: private replay,
a target network, Double-DQN action selection, Huber-clipped TD errors, and a
bounded update norm. Each genome initializes the live and target Q matrices.
All learned Q values, replay samples, target state, and update counters are
discarded at reproduction.

This is a small per-organism implementation of the DQN mechanism, not RLlib's
distributed training runtime. Sharing one RLlib learner or replay buffer across
organisms would violate the experiment's lifetime-learning boundary.

The selected high-exploration configuration beat REINFORCE on short held-out
runs and survived a three-seed 3,000-step gate. However, its learning-off twin
also beat REINFORCE, and the incremental benefit of replay learning was not
detected. It is therefore a promising policy/exploration condition, but not yet
a validated alternative learning algorithm. Ecology, exploration, and replay
use independent deterministic random streams so replay sampling itself cannot
perturb the ecology control. A subsequent exploration-matched parameter search
also failed on confirmation seeds: enabled DQN averaged 177.5 agents versus 176.8
with learning disabled (`p=0.946`). A subsequent untouched gate showed a
positive but inconclusive difference, 182.6 versus 155.1 (`p=0.200`). See
`RESULTS.md`.
