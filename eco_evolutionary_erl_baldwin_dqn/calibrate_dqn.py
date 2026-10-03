"""Initialization-seed-matched calibration runner for lifetime linear Double DQN."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.config import config_erl as base_cfg
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.world import ErlWorld as BaseWorld
from .config import config_dqn
from .world import DqnWorld

def fl(s): return tuple(float(x) for x in s.split(","))
def il(s): return tuple(int(x) for x in s.split(","))
def specs(alphas,epsilons,targets,gammas=(1.0,)):
    yield {"name":"reinforce","algorithm":"reinforce","strategy":"ERL"}
    yield {"name":"evolution_only","algorithm":"reinforce","strategy":"E"}
    for e in epsilons:
        yield {"name":f"dqn_learning_off_e{e:g}","algorithm":"dqn","strategy":"E","alpha":alphas[0],"epsilon":e,"target":targets[0],"gamma":gammas[0]}
    for a in alphas:
        for e in epsilons:
            for g in gammas:
                for t in targets: yield {"name":f"dqn_a{a:g}_e{e:g}_g{g:g}_t{t}","algorithm":"dqn","strategy":"ERL","alpha":a,"epsilon":e,"target":t,"gamma":g}

def run(spec,seed,steps):
    if spec["algorithm"]=="dqn":
        cfg=dict(config_dqn); cfg.update(seed=seed,strategy=spec["strategy"],dqn_alpha=spec["alpha"],dqn_epsilon=spec["epsilon"],dqn_gamma=spec["gamma"],dqn_target_update_interval=spec["target"]); world=DqnWorld(cfg,np.random.default_rng(seed))
    else:
        cfg=dict(base_cfg); cfg.update(seed=seed,strategy=spec["strategy"]); world=BaseWorld(cfg,np.random.default_rng(seed))
    pops=[world.population_counts()["agent"]]; carn=[world.population_counts()["carnivore"]]
    extinct=None; failure=""; td=upd=qmax=float("nan")
    try:
        for completed in range(1,steps+1):
            world.step(); c=world.population_counts(); pops.append(c["agent"]); carn.append(c["carnivore"])
            if isinstance(world,DqnWorld) and (completed%50==0 or c["agent"]==0):
                alive=[a for a in world.agents if a.alive]
                if alive:
                    td=float(np.nanmax([td,max(a.last_td_absmean for a in alive)])); upd=float(np.nanmax([upd,max(a.last_update_norm for a in alive)])); qmax=float(np.nanmax([qmax,max(max(np.max(np.abs(a.action_weights)),np.max(np.abs(a.action_bias))) for a in alive)]))
            if c["agent"]==0: extinct=world.current_step; break
    except (FloatingPointError,OverflowError) as exc: failure=f"{type(exc).__name__}: {exc}"
    if extinct is not None: pops += [0]*(steps+1-len(pops))
    alive=[a for a in world.agents if a.alive]
    disp=float(np.mean([np.linalg.norm(np.r_[(a.action_weights-a.genome.action_weights).ravel(),a.action_bias-a.genome.action_bias]) for a in alive])) if alive and not failure else float("nan")
    return {"treatment":spec["name"],"algorithm":spec["algorithm"],"alpha":spec.get("alpha",float("nan")),"epsilon":spec.get("epsilon",float("nan")),"gamma":spec.get("gamma",float("nan")),"target_interval":spec.get("target",float("nan")),"seed":seed,"steps":steps,"extinction_step":extinct if extinct is not None else "","population_end":pops[-1] if not failure else float("nan"),"population_mean":float(np.mean(pops)) if not failure else float("nan"),"population_peak":max(pops) if not failure else float("nan"),"carnivore_peak":max(carn) if not failure else float("nan"),"births":world._next_agent_id-cfg["n_initial_agents"],"learning_displacement":disp,"td_absmean_sampled_max":td,"update_sampled_max":upd,"q_absmax_sampled":qmax,"numerical_failure":failure}

def summary(rows):
    out=[]
    for t in dict.fromkeys(r["treatment"] for r in rows):
        g=[r for r in rows if r["treatment"]==t and not r["numerical_failure"]]
        out.append({"treatment":t,"n":len(g),"extinctions":sum(r["extinction_step"]!="" for r in g),"population_end_mean":float(np.mean([r["population_end"] for r in g])),"population_mean_mean":float(np.mean([r["population_mean"] for r in g])),"population_peak_mean":float(np.mean([r["population_peak"] for r in g])),"carnivore_peak_mean":float(np.mean([r["carnivore_peak"] for r in g]))})
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--steps",type=int,default=300); p.add_argument("--seeds",type=il,default=(41,42,43)); p.add_argument("--alphas",type=fl,default=(.001,.005,.02)); p.add_argument("--epsilons",type=fl,default=(.05,.1)); p.add_argument("--gammas",type=fl,default=(1.0,)); p.add_argument("--targets",type=il,default=(10,50)); p.add_argument("--out",type=Path,required=True); a=p.parse_args()
    rows=[]
    for seed in a.seeds:
        for spec in specs(a.alphas,a.epsilons,a.targets,a.gammas):
            row=run(spec,seed,a.steps); rows.append(row); print(json.dumps(row,sort_keys=True),flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    a.out.with_suffix(".summary.json").write_text(json.dumps(summary(rows),indent=2)+"\n")
if __name__=="__main__": main()
