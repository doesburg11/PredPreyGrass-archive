# ERL Baldwin with lifetime-private linear PPO

Each organism owns a linear softmax actor, a linear value baseline, and a short
on-policy rollout buffer. Updates use GAE, PPO ratio clipping, entropy
regularization, bounded actor/critic steps, and multiple epochs over each
rollout. The actor starts from the genome; critic, rollout, and all acquired
parameters are discarded at reproduction.

Policy sampling uses a deterministic RNG stream separate from ecology so the
learning-off control is not perturbed by learner bookkeeping.

The selected setting (`actor_alpha=0.01`, `critic_beta=0.01`, `gamma=0.5`,
rollout 4, four epochs) showed a near-signal in its first untouched gate but no
benefit in an independent replication. PPO is therefore not validated as an
alternative lifetime learner. See `RESULTS.md`.
