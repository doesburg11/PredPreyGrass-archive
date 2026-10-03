"""Linear Double-DQN primitives with Huber-clipped TD updates."""

import numpy as np


def q_values(obs, weights, bias):
    values = obs @ weights + bias
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("non-finite DQN values")
    return values


def epsilon_greedy(values, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(0, len(values)))
    maxima = np.flatnonzero(values == np.max(values))
    return int(maxima[int(rng.integers(0, len(maxima)))])


def dqn_batch_update(weights, bias, target_weights, target_bias, batch, *, alpha,
                     gamma, huber_delta, max_update_norm):
    """Apply one averaged semi-gradient Double-DQN update and return diagnostics."""
    delta_w = np.zeros_like(weights); delta_b = np.zeros_like(bias); errors = []
    for obs, action, reward, next_obs, terminal in batch:
        current = float(q_values(obs, weights, bias)[action])
        if terminal:
            target = reward
        else:
            next_action = int(np.argmax(q_values(next_obs, weights, bias)))
            target = reward + gamma * float(q_values(next_obs, target_weights, target_bias)[next_action])
        error = target - current
        if not np.isfinite(error):
            raise FloatingPointError("non-finite DQN TD error")
        errors.append(error)
        clipped = float(np.clip(error, -huber_delta, huber_delta))
        delta_w[:, action] += clipped * obs
        delta_b[action] += clipped
    with np.errstate(over="raise", invalid="raise"):
        delta_w *= alpha / len(batch); delta_b *= alpha / len(batch)
        norm = float(np.linalg.norm(np.r_[delta_w.ravel(), delta_b]))
        if not np.isfinite(norm): raise FloatingPointError("non-finite DQN update")
        scale = 1.0 if norm <= max_update_norm else max_update_norm / norm
        weights += scale * delta_w; bias += scale * delta_b
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(bias)):
        raise FloatingPointError("DQN update produced non-finite parameters")
    return float(np.mean(np.abs(errors))), norm * scale
