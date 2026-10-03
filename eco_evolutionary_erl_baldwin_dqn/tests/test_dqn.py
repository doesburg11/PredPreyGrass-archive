import numpy as np
import pytest

import eco_evolutionary_erl_baldwin_dqn.world as dqn_world_module
from eco_evolutionary_erl_baldwin_dqn.calibrate_dqn import specs
from eco_evolutionary_erl_baldwin_dqn.config import config_dqn
from eco_evolutionary_erl_baldwin_dqn.networks import dqn_batch_update, epsilon_greedy
from eco_evolutionary_erl_baldwin_dqn.world import DqnWorld


def small_config(**overrides):
    cfg = dict(config_dqn); cfg.update(grid_size=12, n_initial_agents=1,
        n_initial_carnivores=0, min_plants=0, min_trees=0, wall_interior_density=0.0,
        plant_growth_prob=0.0, tree_birth_prob=0.0, tree_death_prob=0.0,
        carnivore_spawn_interval=10**9, mutation_rate=0.0); cfg.update(overrides); return cfg


def test_double_dqn_uses_live_argmax_and_target_value():
    w=np.zeros((1,2)); b=np.array([0.0,0.0]); tw=np.zeros((1,2)); tb=np.array([2.0,5.0])
    w[0]=[3.0,1.0]  # live argmax is action 0; target value there is 2, not target max 5
    td,norm=dqn_batch_update(w,b,tw,tb,[(np.array([1.0]),1,1.0,np.array([1.0]),False)],
        alpha=0.1,gamma=0.5,huber_delta=10,max_update_norm=10)
    assert td == pytest.approx(1.0 + 1.0 - 1.0)
    assert b[1] == pytest.approx(0.1)
    assert norm > 0


def test_terminal_has_no_bootstrap():
    w=np.zeros((1,2)); b=np.zeros(2); tw=np.ones((1,2))*100; tb=np.ones(2)*100
    td,_=dqn_batch_update(w,b,tw,tb,[(np.array([1.0]),0,-2.0,None,True)],
        alpha=.1,gamma=1,huber_delta=10,max_update_norm=10)
    assert td == 2.0
    assert b[0] == pytest.approx(-0.2)


def test_update_cap_and_huber_clip():
    w=np.zeros((2,2)); b=np.zeros(2); tw=w.copy(); tb=b.copy()
    _,norm=dqn_batch_update(w,b,tw,tb,[(np.ones(2),0,1e6,None,True)],
        alpha=1,gamma=1,huber_delta=1,max_update_norm=.1)
    assert norm == pytest.approx(.1)


def test_epsilon_greedy_extremes():
    rng=np.random.default_rng(1); values=np.array([0.,10.,0.,0.])
    assert all(epsilon_greedy(values,0,rng)==1 for _ in range(20))
    actions={epsilon_greedy(values,1,rng) for _ in range(100)}
    assert actions == {0,1,2,3}


def test_newborn_discards_live_q_target_and_replay():
    world=DqnWorld(small_config(),np.random.default_rng(2)); parent=world.agents[0]
    parent.action_weights += 9; parent.target_weights += 8; parent.replay.append((1,2,3,4,False))
    parent.energy=world.cfg["reproduction_energy_threshold_agent"]+1; world._handle_agent_reproduction()
    child=max(world.agents,key=lambda a:a.agent_id)
    assert np.array_equal(child.action_weights,child.genome.action_weights)
    assert np.array_equal(child.target_weights,child.genome.action_weights)
    assert len(child.replay)==0


def test_replay_training_and_target_sync():
    world=DqnWorld(small_config(dqn_learning_starts=1,dqn_batch_size=1,dqn_target_update_interval=1),np.random.default_rng(3))
    agent=world.agents[0]; before=agent.action_weights.copy()
    transition=(np.ones(world.obs_dim),0,1.0,np.zeros(world.obs_dim),False)
    world._remember_and_train(agent,transition)
    assert not np.array_equal(agent.action_weights,before)
    assert np.array_equal(agent.target_weights,agent.action_weights)


def test_kill_stores_one_terminal_transition():
    world=DqnWorld(small_config(dqn_learning_starts=64),np.random.default_rng(4)); agent=world.agents[0]
    agent.prev_obs=world._observe_agent(agent); agent.prev_action=0; agent.prev_eval=0.0
    world._kill_agent(agent); world._kill_agent(agent)
    assert len(agent.replay)==1 and agent.replay[0][4] is True


def test_death_is_committed_if_terminal_training_raises(monkeypatch):
    world=DqnWorld(small_config(dqn_learning_starts=1,dqn_batch_size=1),np.random.default_rng(5)); agent=world.agents[0]
    agent.prev_obs=world._observe_agent(agent); agent.prev_action=0; agent.prev_eval=0.0
    deaths=[]; world.on_agent_death=lambda dead, step: deaths.append((dead,step))

    def fail_update(*args, **kwargs):
        raise FloatingPointError("test failure")

    monkeypatch.setattr(dqn_world_module,"dqn_batch_update",fail_update)
    with pytest.raises(FloatingPointError,match="test failure"):
        world._kill_agent(agent)
    assert not agent.alive
    assert (agent.row,agent.col) not in world.occupant
    assert deaths == [(agent,world.current_step)]


def test_learning_off_keeps_q_target_and_replay_unchanged():
    world=DqnWorld(small_config(strategy="E",dqn_learning_starts=1,dqn_batch_size=1),np.random.default_rng(6)); agent=world.agents[0]
    live_before=agent.action_weights.copy(); target_before=agent.target_weights.copy()
    for _ in range(3): world.step()
    assert np.array_equal(agent.action_weights,live_before)
    assert np.array_equal(agent.target_weights,target_before)
    assert len(agent.replay)==0 and agent.learning_updates==0


def test_replay_sampling_does_not_advance_ecology_or_exploration_rng():
    seed=7; cfg=small_config(seed=seed,dqn_learning_starts=1,dqn_batch_size=1)
    trained=DqnWorld(cfg,np.random.default_rng(seed)); control=DqnWorld(cfg,np.random.default_rng(seed))
    transition=(np.ones(trained.obs_dim),0,1.0,np.zeros(trained.obs_dim),False)
    trained._remember_and_train(trained.agents[0],transition)
    assert trained.rng.random() == control.rng.random()
    assert trained.exploration_rng.random() == control.exploration_rng.random()


def test_exploration_matched_grid_shape_and_unique_names():
    grid=list(specs((.005,.02,.05),(.75,),(5,10,50),(.5,.9,1.0)))
    learning=[x for x in grid if x.get("strategy")=="ERL" and x.get("algorithm")=="dqn"]
    off=[x for x in grid if x.get("strategy")=="E" and x.get("algorithm")=="dqn"]
    assert len(grid)==30 and len(learning)==27 and len(off)==1
    assert len({x["name"] for x in grid})==len(grid)


def test_learning_on_off_match_before_first_update():
    seed=8
    common=small_config(seed=seed,dqn_learning_starts=64)
    enabled=DqnWorld(dict(common,strategy="ERL"),np.random.default_rng(seed))
    disabled=DqnWorld(dict(common,strategy="E"),np.random.default_rng(seed))
    for _ in range(5):
        enabled.step(); disabled.step()
        assert enabled.population_counts()==disabled.population_counts()
        assert [(a.agent_id,a.row,a.col,a.energy,a.health) for a in enabled.agents] == [
            (a.agent_id,a.row,a.col,a.energy,a.health) for a in disabled.agents]
