"""Small linear clipped-PPO update used independently by each organism."""

import numpy as np

from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.networks import action_probs


def state_value(obs, weights, bias):
    return float(obs @ weights + bias[0])


def _cap(delta_w, delta_b, maximum):
    norm=float(np.linalg.norm(np.r_[delta_w.ravel(),np.asarray(delta_b).ravel()]))
    if not np.isfinite(norm): raise FloatingPointError("non-finite PPO update")
    scale=1.0 if norm<=maximum or norm==0 else maximum/norm
    return delta_w*scale,delta_b*scale,norm*scale


def ppo_update(actor_weights,actor_bias,critic_weights,critic_bias,transitions,*,
               actor_alpha,critic_beta,gamma,gae_lambda,clip_ratio,entropy_coef,
               epochs,actor_max_update_norm,critic_max_update_norm):
    """Run full-buffer PPO epochs; old log-probabilities remain frozen."""
    if not transitions: return 0.0,0.0,0.0
    values=np.array([state_value(t[0],critic_weights,critic_bias) for t in transitions])
    next_values=np.array([0.0 if t[5] else state_value(t[4],critic_weights,critic_bias) for t in transitions])
    rewards=np.array([t[3] for t in transitions]); terminals=np.array([t[5] for t in transitions],dtype=bool)
    deltas=rewards+gamma*next_values*(~terminals)-values
    advantages=np.zeros(len(transitions)); running=0.0
    for i in range(len(transitions)-1,-1,-1):
        running=deltas[i]+gamma*gae_lambda*(not terminals[i])*running
        advantages[i]=running
    returns=advantages+values
    if len(advantages)>1 and advantages.std()>1e-12:
        advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
    max_actor=max_critic=0.0
    for _ in range(epochs):
        aw=np.zeros_like(actor_weights); ab=np.zeros_like(actor_bias)
        cw=np.zeros_like(critic_weights); cb=np.zeros_like(critic_bias)
        for transition,adv,target in zip(transitions,advantages,returns):
            obs,action,old_logp,_,_,_=transition
            probs=action_probs(obs,actor_weights,actor_bias)
            logp=float(np.log(max(probs[action],1e-12))); ratio=float(np.exp(logp-old_logp))
            clipped=(adv>=0 and ratio>1+clip_ratio) or (adv<0 and ratio<1-clip_ratio)
            if not clipped:
                score=-probs; score[action]+=1.0
                grad=ratio*adv*score
                aw+=np.outer(obs,grad); ab+=grad
            if entropy_coef:
                entropy=-float(np.sum(probs*np.log(np.maximum(probs,1e-12))))
                entropy_grad=probs*(-np.log(np.maximum(probs,1e-12))-entropy)
                aw+=entropy_coef*np.outer(obs,entropy_grad); ab+=entropy_coef*entropy_grad
            error=target-state_value(obs,critic_weights,critic_bias)
            cw+=error*obs; cb[0]+=error
        n=len(transitions)
        daw,dab,an=_cap(actor_alpha*aw/n,actor_alpha*ab/n,actor_max_update_norm)
        dcw,dcb,cn=_cap(critic_beta*cw/n,critic_beta*cb/n,critic_max_update_norm)
        actor_weights+=daw; actor_bias+=dab; critic_weights+=dcw; critic_bias+=dcb
        max_actor=max(max_actor,an); max_critic=max(max_critic,cn)
        if not all(np.all(np.isfinite(x)) for x in (actor_weights,actor_bias,critic_weights,critic_bias)):
            raise FloatingPointError("PPO update produced non-finite parameters")
    return float(np.mean(np.abs(deltas))),max_actor,max_critic
