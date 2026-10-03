# PredPreyGrass-archive

Closed or discontinued experiment modules pulled out of
[PredPreyGrass](https://github.com/doesburg11/PredPreyGrass) to keep that repo
uncluttered. Each is kept for the record: what was tried, what it found, and why
it was stopped. Some reached replicated null conclusions; others stopped at an
earlier falsification or feasibility gate. The original archive set was
transferred with per-module history via `git subtree split`. Later additions
retain their available source, tests, documentation, and recorded result
artifacts; DQN and PPO entered this archive directly from uncommitted source-repo
work, so their provenance is the archive commit that adds them rather than prior
source-repository commits.

This is not simply "code that didn't work": the mechanisms were implemented and
tested, while the effects or feasibility conditions they were built to examine
did not clear their respective gates. Consult each module's README and, where
present, `RESULTS.md` before interpreting its status.

## What's here

### ERL Baldwin alternative lifetime learners (closed 2026-10-03)

Five replacements for the positive ERL Baldwin experiment's existing
REINFORCE-style lifetime update were implemented and tested. All preserve the
Darwinian/Baldwinian boundary: acquired parameters remain private to one life
and reproduction copies only the genome. None produced a reproducible
learning-specific advantage, so the original learner remains the only validated
option. The consolidated comparison is in the main repository's
[`eco_evolutionary_erl_baldwin/README.md`](https://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/evolutionary/eco_evolutionary_erl_baldwin/README.md#comparative-lifetime-learning-investigation-closed-2026-10-03).

These five packages reuse the active baseline ERL Baldwin world. To rerun them,
install the main `PredPreyGrass` project in the Python environment, run commands
from this archive repository's root, and use the top-level package names shown
in each archived README/RESULTS file (for example,
`python -m eco_evolutionary_erl_baldwin_ppo.calibrate_ppo ...`).

- [`eco_evolutionary_erl_baldwin_hebbian`](eco_evolutionary_erl_baldwin_hebbian/) — reward-modulated Hebbian plasticity; failed the controlled single-lifetime gate, including the uniform-positive-alpha control.
- [`eco_evolutionary_erl_baldwin_sarsa`](eco_evolutionary_erl_baldwin_sarsa/) — linear SARSA; failed the prospective 10-seed safety gate with 4/10 extinctions versus 0/10 for REINFORCE.
- [`eco_evolutionary_erl_baldwin_actor_critic`](eco_evolutionary_erl_baldwin_actor_critic/) — linear TD actor-critic; significantly underperformed REINFORCE on the held-out block.
- [`eco_evolutionary_erl_baldwin_dqn`](eco_evolutionary_erl_baldwin_dqn/) — linear Double DQN with replay and target network; stable, but unable to beat its exploration-matched learning-off control reliably.
- [`eco_evolutionary_erl_baldwin_ppo`](eco_evolutionary_erl_baldwin_ppo/) — lifetime-private clipped PPO; a near-signal in the first untouched block failed independent replication.

### [`eco_evolutionary`](eco_evolutionary/) — the family's baseline
The foundational scaffold every other `eco_evolutionary_*` module was cloned from: built on `lineage_rewards`'s
lifecycle tracking (never-reused IDs, parent-child tracking, lineage logs, age limits), adding the first explicit
heritable genome. One trait, `speed`, gates distance-1 vs. distance-2 movement, with superlinear movement cost;
deliberately Baldwinian, not Lamarckian — the genome only affects which actions are physically possible, never the
learned PPO policy weights themselves, which stay shared and are never copied parent-to-offspring.

Across 3 documented training runs it found a real but messy, unconfirmed signal: population-density tuning mattered a
lot (overcrowding erased any speed advantage until grass regrowth was halved and a prey age cap added); with that
fixed, prey speed drifted up then partially reverted, predator speed showed the more persistent upward drift, and
there's a tentative, single-run "Red Queen"-style alternating pattern (prey selection emerges once predators get
effective, then predator selection catches up) that the module's own writeup explicitly flags as not yet a strong
result — it needs sustained reciprocal escalation or repeated cycling across multiple seeds to be more than
suggestive. The writeup's own conclusion recommended `eco_evolutionary_cadence` as the stronger next experiment (a
graded movement-cooldown instead of a hard distance threshold, easier to track); that module was built next and
promptly rejected outright (see below) — the entire documented lineage descending from this module (`cadence` →
`metabolic_rate` → ...) is now archived here, and nothing still active in the main repo imports it, depends on it, or
cites it as a direct clone-parent. Kept in the main repo for a while as a narrative anchor ("baseline of the family")
even after that; archived once it was confirmed nothing still there actually depends on it.

### [`eco_evolutionary_cultural_plasticity`](eco_evolutionary_cultural_plasticity/) — Trial 8
Gene-culture coevolution (dual inheritance): a heritable `plasticity` gene sets how readily an agent's socially-learned
`dialect` trait tracks the local population, on the theory that a *structurally different* mechanism (not another
single-scalar trait) might route around the flat, no-fitness-correlation pattern seen in 7 earlier single-channel trait
trials. **Stopped early** after 3 real seeds: `plasticity_mean` stayed within about one founder-std of its starting
value in all 3, no consistent direction, and `plasticity_repro_spearman` was essentially zero in every seed, both
species — the same flat pattern the dual-inheritance design was meant to avoid. The ~21h neutral-control leg was not
run once the real seeds showed no sign of a mechanism-level improvement.

### [`eco_evolutionary_cultural_plasticity_seasonal`](eco_evolutionary_cultural_plasticity_seasonal/) — Trial 9
Follow-up to Trial 8, testing Rogers' Paradox (Rogers 1988) directly: the *target* dialect flips over time (a seasonal
cycle), so social learning should only pay off if agents can also detect that the environment has changed and it's
worth re-learning — the condition Rogers' Paradox says social learning fails without. Seed 42 ran to completion
(1000/1000 iterations, 2026-08-08): same flat result as Trial 8. Consistent with Rogers' Paradox actually being the
explanation for both trials' null results, not a design flaw specific to either one. (This module's own in-tree README
status box was never updated after that run and still says "not yet run" — trust this summary and Trial 8/9's entries
in `predpreygrass/evolutionary/RESULTS.md` on the main repo over that stale box.)

### [`eco_evolutionary_nuptial_gift`](eco_evolutionary_nuptial_gift/)
Obligate male-provisioning: sexed predators, males give a nuptial gift to females, tested whether that changes
reproductive dynamics. **Stopped early, 2026-08-03**, not completed. One real seed (42) ran to full completion
(1000/1000 iterations, 15.9h): too few reproductions occurred to produce a meaningful signal. The remaining seeds and
the neutral-control leg (planned: 3 real + 3 neutral-control seeds, ~4 more days of compute) were not run once that
became clear. A fixed-genome two-point fitness sweep (`0.0` vs `1.0`) had already confirmed a real, dramatic fitness
landscape exists around this trait — the gap is in observing it emerge under evolution with too few births to sample,
not in the mechanism being inert.

### [`eco_evolutionary_cadence`](eco_evolutionary_cadence/) — Trial 1
A `speed` genome that controls movement *frequency* rather than movement distance: an agent
only gets a real move on roughly 1-in-6 steps depending on its evolved cadence. **Rejected.**
Confirmed directly to structurally prevent a sustainable predator population regardless of
policy quality — predators went extinct in 30/30 sampled seeds under a trained policy. The
mechanic itself was the problem, not something tunable away. Abandoned for
`eco_evolutionary_metabolic_rate` (kept in the main repo), which already had partial evidence
of a working sustainability loop.

### [`eco_evolutionary_cooperation`](eco_evolutionary_cooperation/) — Trial 5
`cooperation_rate`: a heritable fraction of *that step's* catch/graze energy donated to
same-species neighbors, founder mean 0.0 (identical to no-genome baseline, so any positive
drift is a direct selection signal). Motivated by the Baldwin-effect/cooperation literature
link (Suzuki & Arita; Taylor 1992's kin-competition cancellation result). **Paused after Pilot
1, not replicated further (2026-07-18).** The neutral-drift control drifted as much as or more
than the real run in both species; with `metabolic_rate` and `investment` already confirmed
null via proper 3-seed replication, a full replication here would most likely have been a
third data point confirming an already-established pattern, not new information.

### [`eco_evolutionary_metabolic_code`](eco_evolutionary_metabolic_code/) — Trial 7
A length-10 combinatorial genome (CORRECT/WRONG/PLASTIC per locus, Hinton & Nowlan 1987
needle-in-haystack design) instead of a smooth scalar — a genuine attempt to give evolution a
landscape that needs learning to find, per the theoretical note in the main repo's
`predpreygrass/evolutionary/RESULTS.md`. **Complete, null — and reversed on the headline
metric**, not merely flat. The structurally different design didn't rescue the pattern the
smooth-scalar trials were already showing.

### [`eco_evolutionary_metabolic_rate`](eco_evolutionary_metabolic_rate/) — Trial 3
`metabolic_rate`: sub-linear energy gain vs. linear cost, creating a policy-dependent interior
optimum. This is where the project's rigorous drift-vs-control replication methodology was
actually built, iteration by iteration (population-ratio caps rejected as biologically
unmotivated; a Holling-type individual satiation throttle adopted instead; a neutral-drift
control introduced). A promising single-run read for prey **did not survive a proper 3-seed
real-vs-control replication (Mann-Whitney U) — null for both species.**

### [`eco_evolutionary_investment`](eco_evolutionary_investment/) — Trial 6
`offspring_investment_fraction`: how much energy a parent hands each offspring at birth — a
real tradeoff curve (R6 confirmed a genuine fixed-genome fitness landscape exists, ruling out
"there's no signal to detect" as the null explanation). Went through more replication rounds
than any other trait in the family precisely because it kept looking like the one real
exception: R9's population-scaled run found prey's 3-vs-3 real-vs-control separation hit the
exact statistical ceiling n=3 can produce (U=9, p=0.050) while predator stayed null — directly
suggestive, not yet confirmed. **R10 extended prey to n=6 (3 more real + 3 more control seeds)
specifically to settle that question, and it reversed instead of strengthening**: p moved from
0.050 up to 0.120, the textbook signature of a small-sample artifact rather than a real,
merely-underpowered effect. An independent Hunt (2006) full-trajectory model-fit agreed,
finding pure drift (unbiased random walk) in all 24 real+control seed/species trajectories
across both rounds. Closed as the last data point in the single-continuous-scalar-trait family
to be settled, with no exception surviving replication after all.

## Why these fourteen specifically

Judged against the rest of the `eco_evolutionary_*` family (see the main repo's `predpreygrass/evolutionary/README.md`
and `RESULTS.md`): these fourteen reached a genuine stop with nothing positive to build on, unlike (for example)
`eco_evolutionary_metabolic_rate_positive_control`, which is also a null/weak result but stayed in the main repo
because it directly informed the design of `eco_evolutionary_erl_baldwin` (the project's strongest result, p<0.00001,
n=100/condition) — it has real stepping-stone value that these don't. `eco_evolutionary_erl_flagship` (closed, negative,
but thorough) was considered and deliberately kept in the main repo rather than archived alongside these.

## Recovering full context

Each module's own `README.md`/`RESULTS.md` here has the details. For the cross-trial narrative (how each trial's null
result shaped the next module's design), see `predpreygrass/evolutionary/RESULTS.md` in the main PredPreyGrass repo,
which is retained there even though the module code has moved here.
