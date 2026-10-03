"""Baseline ecology with private, per-lifetime linear clipped PPO."""

from dataclasses import dataclass,field
import numpy as np

from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.genome import crossover,founder_genome,mutate
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.networks import action_probs,evaluate,sample_action
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.world import Agent,ErlWorld as BaselineWorld,N_ACTIONS
from .networks import ppo_update

LEARNING_STRATEGIES=frozenset(("ERL","L","ERLC","ERLK","ERLS"))


@dataclass
class PpoAgent(Agent):
    critic_weights: np.ndarray|None=None
    critic_bias: np.ndarray|None=None
    rollout: list=field(default_factory=list)
    prev_logp: float=0.0
    learning_updates: int=0
    last_td_absmean: float=0.0
    last_actor_update_norm: float=0.0
    last_critic_update_norm: float=0.0


class PpoWorld(BaselineWorld):
    def __init__(self,config,rng):
        self._validate(config)
        seed=int(config["seed"])
        seed_seq=getattr(rng.bit_generator,"_seed_seq",None)
        if seed_seq is None or seed_seq.entropy!=seed:
            raise ValueError("PpoWorld requires rng initialized from config['seed']")
        self.policy_rng=np.random.default_rng(np.random.SeedSequence([seed,0xA110]))
        super().__init__(config,rng)

    @staticmethod
    def _validate(c):
        for k in ("ppo_actor_alpha","ppo_critic_beta","ppo_actor_max_update_norm","ppo_critic_max_update_norm"):
            if not(np.isfinite(c[k]) and c[k]>0): raise ValueError(f"{k} must be finite and > 0")
        for k in ("ppo_gamma","ppo_gae_lambda"):
            if not(np.isfinite(c[k]) and 0<=c[k]<=1): raise ValueError(f"{k} must be in [0, 1]")
        if not(np.isfinite(c["ppo_clip_ratio"]) and c["ppo_clip_ratio"]>0): raise ValueError("invalid PPO clip")
        if not(np.isfinite(c["ppo_entropy_coef"]) and c["ppo_entropy_coef"]>=0): raise ValueError("invalid PPO entropy")
        if not np.isfinite(c["ppo_terminal_bonus"]): raise ValueError("invalid PPO terminal bonus")
        for k in ("ppo_rollout_length","ppo_epochs"):
            if not isinstance(c[k],int) or c[k]<=0: raise ValueError(f"{k} must be a positive integer")

    def _make_agent(self,g,row,col,generation):
        return PpoAgent(agent_id=self._next_agent_id,row=row,col=col,energy=self.cfg["initial_energy_agent"],
            health=self.cfg["initial_health_agent"],in_tree=False,genome=g,action_weights=g.action_weights.copy(),
            action_bias=g.action_bias.copy(),generation=generation,born_step=self.current_step,
            critic_weights=np.zeros(self.obs_dim),critic_bias=np.zeros(1))

    def _spawn_founder_agent(self):
        cell=self._random_empty_cell()
        if cell is None:return
        g=founder_genome(self.obs_dim,N_ACTIONS,self.rng,self.cfg["founder_weight_std"],fixed_eval_weights=self.cfg.get("fixed_eval_weights"))
        a=self._make_agent(g,*cell,0);self._next_agent_id+=1;self.agents.append(a);self.occupant[cell]=a

    def _train(self,a):
        a.last_td_absmean,a.last_actor_update_norm,a.last_critic_update_norm=ppo_update(
            a.action_weights,a.action_bias,a.critic_weights,a.critic_bias,a.rollout,
            actor_alpha=self.cfg["ppo_actor_alpha"],critic_beta=self.cfg["ppo_critic_beta"],
            gamma=self.cfg["ppo_gamma"],gae_lambda=self.cfg["ppo_gae_lambda"],clip_ratio=self.cfg["ppo_clip_ratio"],
            entropy_coef=self.cfg["ppo_entropy_coef"],epochs=self.cfg["ppo_epochs"],
            actor_max_update_norm=self.cfg["ppo_actor_max_update_norm"],critic_max_update_norm=self.cfg["ppo_critic_max_update_norm"])
        a.learning_updates+=1;a.rollout.clear()

    def _close_transition(self,a,reward,next_obs,terminal):
        a.rollout.append((a.prev_obs.copy(),a.prev_action,a.prev_logp,reward,None if terminal else next_obs.copy(),terminal))
        if terminal or len(a.rollout)>=self.cfg["ppo_rollout_length"]: self._train(a)

    def _step_agents(self):
        order=list(self.agents);self.rng.shuffle(order);learning=self.strategy in LEARNING_STRATEGIES;coop=self.strategy in ("C","ERLC")
        for a in order:
            if not a.alive:continue
            obs=self._observe_agent(a);ev=evaluate(obs,a.genome.eval_weights,a.genome.eval_bias)
            if learning and a.prev_obs is not None:self._close_transition(a,ev-a.prev_eval,obs,False)
            if self.strategy=="B": probs=None;action=int(self.rng.integers(0,N_ACTIONS));logp=0.0
            else:
                probs=action_probs(obs,a.action_weights,a.action_bias);action=sample_action(probs,self.policy_rng);logp=float(np.log(max(probs[action],1e-12)))
            a.prev_obs=obs;a.prev_action=action;a.prev_eval=ev;a.prev_logp=logp
            self._resolve_agent_action(a,action,track_forage=coop)
            if a.alive:
                a.energy-=self.cfg["basal_energy_cost_agent"]
                if a.energy<=0 or a.health<=0:self._kill_agent(a)

    def _kill_agent(self,a):
        if not a.alive:return
        transition=None
        if self.strategy in LEARNING_STRATEGIES and a.prev_obs is not None:
            obs=self._observe_agent(a);ev=evaluate(obs,a.genome.eval_weights,a.genome.eval_bias)
            transition=(ev-a.prev_eval+self.cfg["ppo_terminal_bonus"],None,True)
        super()._kill_agent(a)
        if transition:self._close_transition(a,*transition)

    def _handle_agent_reproduction(self):
        coop=self.strategy in ("C","ERLC");new=[]
        for a in self.agents:
            if not a.alive:continue
            threshold=self.cfg["reproduction_energy_threshold_agent"]
            if coop and self._agent_group_is_cooperative_fit(a):threshold*=1-self.cfg["coop_threshold_discount_frac"]
            if a.energy<threshold or len(self.agents)+len(new)>=self.cfg["max_population_cap"]:continue
            cell=self._nearest_empty_adjacent(a.row,a.col)
            if cell is None:continue
            mate=None
            if self.strategy in ("L","F"):g=a.genome.copy()
            else:
                mate=self._nearest_mate(a);g=a.genome.copy()
                if mate is not None:g=crossover(a.genome,mate.genome,self.rng)
                g=mutate(g,self.rng,self.cfg["mutation_rate"],self.cfg["mutation_std"])
            self.constraint_tracker.record(a.genome.flatten(),g.flatten());a.energy-=self.cfg["reproduction_energy_cost_agent"];a.offspring_count+=1
            if mate is not None:mate.offspring_count+=1
            if coop:a.last_reproduce_step=self.current_step
            child=self._make_agent(g,*cell,a.generation+1);self._next_agent_id+=1;new.append(child);self.occupant[cell]=child
        self.agents.extend(new)

ErlWorld=PpoWorld
