"""Baseline ecology with private per-lifetime linear Double-DQN learning."""

from collections import deque
from dataclasses import dataclass
import numpy as np

from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.genome import crossover, founder_genome, mutate
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.networks import evaluate
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.world import Agent, ErlWorld as BaselineWorld, N_ACTIONS
from eco_evolutionary_erl_baldwin_dqn.networks import dqn_batch_update, epsilon_greedy, q_values

LEARNING_STRATEGIES = frozenset(("ERL", "L", "ERLC", "ERLK", "ERLS"))


@dataclass
class DqnAgent(Agent):
    target_weights: np.ndarray | None = None
    target_bias: np.ndarray | None = None
    replay: deque | None = None
    learning_updates: int = 0
    last_td_absmean: float = 0.0
    last_update_norm: float = 0.0


class DqnWorld(BaselineWorld):
    def __init__(self, config, rng):
        self._validate(config)
        # Keep learner bookkeeping from advancing the ecology RNG. This makes
        # initialization-seed controls meaningfully comparable until policies
        # choose different actions, rather than diverging merely because replay
        # sampled a minibatch.
        seed = int(config["seed"])
        self.exploration_rng = np.random.default_rng(np.random.SeedSequence([seed, 0xD011]))
        self.replay_rng = np.random.default_rng(np.random.SeedSequence([seed, 0xD012]))
        super().__init__(config, rng)

    @staticmethod
    def _validate(cfg):
        for key in ("dqn_alpha", "dqn_huber_delta", "dqn_max_update_norm"):
            if not (np.isfinite(cfg[key]) and cfg[key] > 0): raise ValueError(f"{key} must be finite and > 0")
        if not (0 <= cfg["dqn_gamma"] <= 1 and 0 <= cfg["dqn_epsilon"] <= 1): raise ValueError("invalid DQN gamma/epsilon")
        for key in ("dqn_replay_capacity", "dqn_batch_size", "dqn_learning_starts", "dqn_target_update_interval"):
            if not isinstance(cfg[key], int) or cfg[key] <= 0: raise ValueError(f"{key} must be a positive integer")
        if cfg["dqn_batch_size"] > cfg["dqn_replay_capacity"] or cfg["dqn_learning_starts"] > cfg["dqn_replay_capacity"]:
            raise ValueError("DQN batch/learning-start threshold exceeds replay capacity")

    def _make_agent(self, genome, row, col, generation):
        return DqnAgent(agent_id=self._next_agent_id, row=row, col=col,
            energy=self.cfg["initial_energy_agent"], health=self.cfg["initial_health_agent"],
            in_tree=False, genome=genome, action_weights=genome.action_weights.copy(),
            action_bias=genome.action_bias.copy(), generation=generation,
            born_step=self.current_step, target_weights=genome.action_weights.copy(),
            target_bias=genome.action_bias.copy(), replay=deque(maxlen=self.cfg["dqn_replay_capacity"]))

    def _spawn_founder_agent(self):
        cell = self._random_empty_cell()
        if cell is None: return
        genome = founder_genome(self.obs_dim, N_ACTIONS, self.rng, self.cfg["founder_weight_std"],
                                fixed_eval_weights=self.cfg.get("fixed_eval_weights"))
        agent = self._make_agent(genome, *cell, 0); self._next_agent_id += 1
        self.agents.append(agent); self.occupant[cell] = agent

    def _remember_and_train(self, agent, transition):
        agent.replay.append(transition)
        if len(agent.replay) < self.cfg["dqn_learning_starts"]: return
        size = min(self.cfg["dqn_batch_size"], len(agent.replay))
        indices = self.replay_rng.choice(len(agent.replay), size=size, replace=False)
        batch = [agent.replay[int(i)] for i in indices]
        agent.last_td_absmean, agent.last_update_norm = dqn_batch_update(
            agent.action_weights, agent.action_bias, agent.target_weights, agent.target_bias, batch,
            alpha=self.cfg["dqn_alpha"], gamma=self.cfg["dqn_gamma"],
            huber_delta=self.cfg["dqn_huber_delta"], max_update_norm=self.cfg["dqn_max_update_norm"])
        agent.learning_updates += 1
        if agent.learning_updates % self.cfg["dqn_target_update_interval"] == 0:
            np.copyto(agent.target_weights, agent.action_weights); np.copyto(agent.target_bias, agent.action_bias)

    def _step_agents(self):
        order = list(self.agents); self.rng.shuffle(order)
        learning = self.strategy in LEARNING_STRATEGIES; coop = self.strategy in ("C", "ERLC")
        for agent in order:
            if not agent.alive: continue
            obs = self._observe_agent(agent); ev = evaluate(obs, agent.genome.eval_weights, agent.genome.eval_bias)
            if learning and agent.prev_obs is not None:
                self._remember_and_train(agent, (agent.prev_obs.copy(), agent.prev_action, ev-agent.prev_eval, obs.copy(), False))
            if self.strategy == "B": action = int(self.rng.integers(0, N_ACTIONS))
            else: action = epsilon_greedy(q_values(obs, agent.action_weights, agent.action_bias), self.cfg["dqn_epsilon"], self.exploration_rng)
            agent.prev_obs = obs; agent.prev_action = action; agent.prev_eval = ev
            self._resolve_agent_action(agent, action, track_forage=coop)
            if agent.alive:
                agent.energy -= self.cfg["basal_energy_cost_agent"]
                if agent.energy <= 0 or agent.health <= 0: self._kill_agent(agent)

    def _kill_agent(self, agent):
        if not agent.alive: return
        terminal_transition = None
        if self.strategy in LEARNING_STRATEGIES and agent.prev_obs is not None:
            final_obs = self._observe_agent(agent); final_ev = evaluate(final_obs, agent.genome.eval_weights, agent.genome.eval_bias)
            reward = final_ev-agent.prev_eval+self.cfg["dqn_terminal_bonus"]
            terminal_transition = (agent.prev_obs.copy(), agent.prev_action, reward, None, True)
        # Preserve the baseline's commit-before-observer guarantee and make death
        # durable even if the terminal learning update rejects non-finite data.
        super()._kill_agent(agent)
        if terminal_transition is not None:
            self._remember_and_train(agent, terminal_transition)

    def _handle_agent_reproduction(self):
        coop = self.strategy in ("C", "ERLC"); newborns = []
        for agent in self.agents:
            if not agent.alive: continue
            threshold = self.cfg["reproduction_energy_threshold_agent"]
            if coop and self._agent_group_is_cooperative_fit(agent): threshold *= 1-self.cfg["coop_threshold_discount_frac"]
            if agent.energy < threshold or len(self.agents)+len(newborns) >= self.cfg["max_population_cap"]: continue
            cell = self._nearest_empty_adjacent(agent.row, agent.col)
            if cell is None: continue
            mate = None
            if self.strategy in ("L", "F"): child_genome = agent.genome.copy()
            else:
                mate = self._nearest_mate(agent); child_genome = agent.genome.copy()
                if mate is not None: child_genome = crossover(agent.genome, mate.genome, self.rng)
                child_genome = mutate(child_genome, self.rng, self.cfg["mutation_rate"], self.cfg["mutation_std"])
            self.constraint_tracker.record(agent.genome.flatten(), child_genome.flatten())
            agent.energy -= self.cfg["reproduction_energy_cost_agent"]; agent.offspring_count += 1
            if mate is not None: mate.offspring_count += 1
            if coop: agent.last_reproduce_step = self.current_step
            child = self._make_agent(child_genome, *cell, agent.generation+1); self._next_agent_id += 1
            newborns.append(child); self.occupant[cell] = child
        self.agents.extend(newborns)


ErlWorld = DqnWorld
