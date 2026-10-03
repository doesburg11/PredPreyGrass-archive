import numpy as np
import pytest
from eco_evolutionary_erl_baldwin_ppo.config import config_ppo
from eco_evolutionary_erl_baldwin_ppo.networks import ppo_update
from eco_evolutionary_erl_baldwin_ppo.world import PpoWorld

def cfg(**kw):
    c=dict(config_ppo);c.update(seed=1,grid_size=12,n_initial_agents=1,n_initial_carnivores=0,min_plants=0,min_trees=0,wall_interior_density=0.,plant_growth_prob=0.,tree_birth_prob=0.,tree_death_prob=0.,carnivore_spawn_interval=10**9,mutation_rate=0.);c.update(kw);return c

def update(w,b,cw,cb,ts,**kw):
    p=dict(actor_alpha=.1,critic_beta=.1,gamma=.9,gae_lambda=.95,clip_ratio=.2,entropy_coef=0.,epochs=1,actor_max_update_norm=10.,critic_max_update_norm=10.);p.update(kw);return ppo_update(w,b,cw,cb,ts,**p)

def test_positive_advantage_increases_selected_logit():
    w=np.zeros((1,2));b=np.zeros(2);cw=np.zeros(1);cb=np.zeros(1);obs=np.ones(1)
    update(w,b,cw,cb,[(obs,0,np.log(.5),1.,None,True)])
    assert b[0]>b[1] and cb[0]>0

def test_clipped_positive_ratio_has_no_actor_gradient():
    w=np.zeros((1,2));b=np.array([2.,0.]);cw=np.zeros(1);cb=np.zeros(1);obs=np.ones(1);before=(w.copy(),b.copy())
    update(w,b,cw,cb,[(obs,0,np.log(.5),1.,None,True)],clip_ratio=.2)
    assert np.array_equal(w,before[0]) and np.array_equal(b,before[1])

def test_terminal_has_no_value_bootstrap():
    w=np.zeros((1,2));b=np.zeros(2);cw=np.array([100.]);cb=np.array([0.]);obs=np.ones(1)
    td,_,_=update(w,b,cw,cb,[(obs,0,np.log(.5),-2.,None,True)])
    assert td==pytest.approx(102.)

def test_rollout_trains_and_clears():
    world=PpoWorld(cfg(ppo_rollout_length=1,ppo_epochs=1),np.random.default_rng(1));a=world.agents[0];before=a.action_weights.copy();world.step()
    assert a.learning_updates==0
    world.step();assert a.learning_updates==1 and not a.rollout and not np.array_equal(a.action_weights,before)

def test_newborn_discards_acquired_state():
    world=PpoWorld(cfg(seed=2),np.random.default_rng(2));a=world.agents[0];a.action_weights+=9;a.critic_weights+=8;a.rollout.append((1,));a.energy=world.cfg["reproduction_energy_threshold_agent"]+1;world._handle_agent_reproduction();child=max(world.agents,key=lambda x:x.agent_id)
    assert np.array_equal(child.action_weights,child.genome.action_weights) and np.all(child.critic_weights==0) and not child.rollout

def test_learning_off_never_fills_rollout_or_updates():
    world=PpoWorld(cfg(seed=3,strategy="E",ppo_rollout_length=1),np.random.default_rng(3));a=world.agents[0];before=a.action_weights.copy()
    for _ in range(3):world.step()
    assert not a.rollout and a.learning_updates==0 and np.array_equal(a.action_weights,before)

def test_rng_seed_must_match_config():
    with pytest.raises(ValueError,match=r"config\['seed'\]"):
        PpoWorld(cfg(seed=4),np.random.default_rng(5))

def test_defaults_are_frozen_candidate():
    assert config_ppo["ppo_actor_alpha"]==.01
    assert config_ppo["ppo_critic_beta"]==.01
    assert config_ppo["ppo_gamma"]==.5
    assert config_ppo["ppo_rollout_length"]==4
    assert config_ppo["ppo_epochs"]==4
