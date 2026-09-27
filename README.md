# PredPreyGrass-archive

Closed, dead-end experiment modules pulled out of [PredPreyGrass](https://github.com/doesburg11/PredPreyGrass) to keep
that repo uncluttered. Each module here ran to a real conclusion — a null result, not an abandoned half-build — and is
kept for the record: what was tried, what it found, and why it was closed. Each subdirectory keeps its own full git
history from PredPreyGrass (via `git subtree split`) and its own README/RESULTS.md with the full numbers.

This is not "code that didn't work" — the mechanisms are implemented and tested correctly; the *biological effect they
were built to demonstrate* did not show up in training. See each module's own `RESULTS.md` for the real-seed numbers
before assuming there's nothing there.

## What's here

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

## Why these seven specifically

Judged against the rest of the `eco_evolutionary_*` family (see the main repo's `predpreygrass/evolutionary/README.md`
and `RESULTS.md`): these seven reached a genuine stop with nothing positive to build on, unlike (for example)
`eco_evolutionary_metabolic_rate_positive_control`, which is also a null/weak result but stayed in the main repo
because it directly informed the design of `eco_evolutionary_erl_baldwin` (the project's strongest result, p<0.00001,
n=100/condition) — it has real stepping-stone value that these don't. `eco_evolutionary_erl_flagship` (closed, negative,
but thorough) was considered and deliberately kept in the main repo rather than archived alongside these.

## Recovering full context

Each module's own `README.md`/`RESULTS.md` here has the details. For the cross-trial narrative (how each trial's null
result shaped the next module's design), see `predpreygrass/evolutionary/RESULTS.md` in the main PredPreyGrass repo,
which is retained there even though the module code has moved here.
