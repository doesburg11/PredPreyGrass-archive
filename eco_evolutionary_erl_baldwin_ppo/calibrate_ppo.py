"""Development and validation runner for lifetime-private linear PPO."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.config import config_erl as base_cfg
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.world import ErlWorld as BaseWorld
from .config import config_ppo
from .world import PpoWorld

def fl(s):return tuple(float(x) for x in s.split(","))
def il(s):return tuple(int(x) for x in s.split(","))
def specs(alphas,betas,gammas,rollouts,epochs):
    yield {"name":"reinforce","algorithm":"reinforce","strategy":"ERL"}
    yield {"name":"evolution_only","algorithm":"reinforce","strategy":"E"}
    yield {"name":"ppo_learning_off","algorithm":"ppo","strategy":"E","alpha":alphas[0],"beta":betas[0],"gamma":gammas[0],"rollout":rollouts[0],"epochs":epochs[0]}
    for a in alphas:
        for b in betas:
            for g in gammas:
                for r in rollouts:
                    for e in epochs:yield {"name":f"ppo_a{a:g}_b{b:g}_g{g:g}_r{r}_e{e}","algorithm":"ppo","strategy":"ERL","alpha":a,"beta":b,"gamma":g,"rollout":r,"epochs":e}

def run(spec,seed,steps):
    if spec["algorithm"]=="ppo":
        cfg=dict(config_ppo);cfg.update(seed=seed,strategy=spec["strategy"],ppo_actor_alpha=spec["alpha"],ppo_critic_beta=spec["beta"],ppo_gamma=spec["gamma"],ppo_rollout_length=spec["rollout"],ppo_epochs=spec["epochs"]);w=PpoWorld(cfg,np.random.default_rng(seed))
    else:
        cfg=dict(base_cfg);cfg.update(seed=seed,strategy=spec["strategy"]);w=BaseWorld(cfg,np.random.default_rng(seed))
    pops=[w.population_counts()["agent"]];failure="";ext=""
    try:
        for _ in range(steps):
            w.step();n=w.population_counts()["agent"];pops.append(n)
            if n==0:ext=w.current_step;pops += [0]*(steps+1-len(pops));break
    except (FloatingPointError,OverflowError) as x:failure=f"{type(x).__name__}: {x}"
    alive=[a for a in w.agents if a.alive]
    disp=float(np.mean([np.linalg.norm(np.r_[(a.action_weights-a.genome.action_weights).ravel(),a.action_bias-a.genome.action_bias]) for a in alive])) if alive and not failure else float("nan")
    return {"treatment":spec["name"],"algorithm":spec["algorithm"],"alpha":spec.get("alpha",float("nan")),"beta":spec.get("beta",float("nan")),"gamma":spec.get("gamma",float("nan")),"rollout":spec.get("rollout",float("nan")),"epochs":spec.get("epochs",float("nan")),"seed":seed,"steps":steps,"extinction_step":ext,"population_end":pops[-1] if not failure else float("nan"),"population_mean":float(np.mean(pops)) if not failure else float("nan"),"population_peak":max(pops) if not failure else float("nan"),"births":w._next_agent_id-cfg["n_initial_agents"],"learning_displacement":disp,"numerical_failure":failure}

def summary(rows):
    out=[]
    for t in dict.fromkeys(x["treatment"] for x in rows):
        g=[x for x in rows if x["treatment"]==t and not x["numerical_failure"]]
        out.append({"treatment":t,"n":len(g),"extinctions":sum(x["extinction_step"]!="" for x in g),"population_mean_mean":float(np.mean([x["population_mean"] for x in g])),"population_end_mean":float(np.mean([x["population_end"] for x in g]))})
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--steps",type=int,default=300);p.add_argument("--seeds",type=il,default=(63,64,65));p.add_argument("--alphas",type=fl,default=(.002,.01));p.add_argument("--betas",type=fl,default=(.01,.05));p.add_argument("--gammas",type=fl,default=(.5,.9));p.add_argument("--rollouts",type=il,default=(4,8));p.add_argument("--epochs",type=il,default=(2,4));p.add_argument("--out",type=Path,required=True);a=p.parse_args();rows=[]
    for seed in a.seeds:
        for s in specs(a.alphas,a.betas,a.gammas,a.rollouts,a.epochs):
            row=run(s,seed,a.steps);rows.append(row);print(json.dumps(row,sort_keys=True),flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    a.out.with_suffix(".summary.json").write_text(json.dumps(summary(rows),indent=2)+"\n")
if __name__=="__main__":main()
