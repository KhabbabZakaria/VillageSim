"""
drl.py
------
Actor-Critic Deep Reinforcement Learning engine.

Each Villager owns one ActorCritic instance.
No external ML framework required — pure NumPy.

Architecture
~~~~~~~~~~~~
    Input  →  Hidden (ReLU)  →  Actor head (softmax policy)
                              →  Critic head (scalar value estimate)

Training
~~~~~~~~
    Advantage Actor-Critic (A2C) with experience replay.
    - Stores (s, a, r, s', done) transitions in a fixed-size circular buffer.
    - On each train() call, runs one gradient-descent pass over a recent batch.
    - Actor loss  : policy-gradient weighted by advantage
    - Critic loss : mean-squared TD error

Knowledge Transfer (reproduction)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    child_brain = parent_a.brain.breed(parent_b.brain)
    - Weight crossover  : each weight independently drawn from either parent
    - Gaussian mutation : ~12% of weights perturbed with small Gaussian noise
    - Memory seeding    : child inherits last N transitions from both parents
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, TYPE_CHECKING

import config as cfg


# ── replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Fixed-size circular buffer for (s, a, r, s', done) experience tuples."""

    def __init__(self, capacity: int = cfg.AC_MAX_MEMORY) -> None:
        self.capacity = capacity
        self._buf: List[dict] = []

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float = 0.0,
    ) -> None:
        self._buf.append(
            dict(s=state, a=action, r=float(reward), ns=next_state, done=done, lp=float(log_prob))
        )
        if len(self._buf) > self.capacity:
            self._buf.pop(0)

    def sample(self, n: int) -> List[dict]:
        """Return the most recent *n* transitions (temporal recency bias)."""
        return self._buf[-n:]

    def sample_random(self, n: int) -> List[dict]:
        """Return up to *n* randomly sampled transitions (no replacement)."""
        n = min(n, len(self._buf))
        if n == 0:
            return []
        indices = np.random.choice(len(self._buf), size=n, replace=False)
        return [self._buf[i] for i in indices]

    def seed(self, transitions: List[dict]) -> None:
        """Pre-populate buffer with knowledge-transferred transitions."""
        for t in transitions:
            self._buf.append(t)
            if len(self._buf) > self.capacity:
                self._buf.pop(0)

    def __len__(self) -> int:
        return len(self._buf)


# ── actor-critic network ──────────────────────────────────────────────────────

class ActorCritic:
    """
    Minimal two-head neural network.

    Parameters
    ----------
    state_size  : dimensionality of the state vector
    action_size : number of discrete actions
    hidden_size : neurons in the single hidden layer
    """

    def __init__(
        self,
        state_size:  int = cfg.AC_STATE_SIZE,
        action_size: int = cfg.AC_ACTION_SIZE,
        hidden_size: int = cfg.AC_HIDDEN_SIZE,
    ) -> None:
        self.ss = state_size
        self.as_ = action_size
        self.hs = hidden_size

        # Weight initialisation (He-style scaling for ReLU)
        scale = np.sqrt(2.0 / state_size)
        self.W1  = (np.random.randn(hidden_size, state_size)  * scale).astype(np.float32)
        self.b1  = np.zeros(hidden_size, dtype=np.float32)
        self.W2a = (np.random.randn(action_size, hidden_size) * 0.3).astype(np.float32)
        self.b2a = np.zeros(action_size, dtype=np.float32)
        self.W2c = (np.random.randn(1, hidden_size)           * 0.3).astype(np.float32)
        self.b2c = np.zeros(1, dtype=np.float32)

        self.lr    = cfg.AC_LEARNING_RATE
        self.gamma = cfg.AC_GAMMA

        self.memory = ReplayBuffer(cfg.AC_MAX_MEMORY)

    # ── forward pass ─────────────────────────────────────────────────────────

    def forward(
        self, state: np.ndarray
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Returns
        -------
        probs  : softmax action probabilities  (shape: action_size,)
        value  : scalar state-value estimate
        hidden : hidden-layer activations (needed for backprop)
        """
        hidden = np.maximum(0.0, self.W1 @ state + self.b1)   # ReLU

        logits = self.W2a @ hidden + self.b2a
        logits -= logits.max()                                  # numerical stability
        exp    = np.exp(logits)
        probs  = exp / exp.sum()

        value = float((self.W2c @ hidden + self.b2c)[0])
        return probs, value, hidden

    # ── action selection ──────────────────────────────────────────────────────

    def sample_action(self, probs: np.ndarray) -> int:
        """Stochastic sampling — used during the *training* phase."""
        return int(np.random.choice(len(probs), p=probs))

    def greedy_action(self, probs: np.ndarray) -> int:
        """Deterministic argmax — used during the *testing* phase."""
        return int(np.argmax(probs))

    # ── store transition ──────────────────────────────────────────────────────

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
        log_prob: float = 0.0,
    ) -> None:
        self.memory.push(state, action, reward, next_state, done, log_prob)

    # ── training step ─────────────────────────────────────────────────────────

    def train(self) -> Optional[dict]:
        """
        Run PPO gradient updates over a recent mini-batch for K epochs.

        Returns a dict of training metrics (losses, mean advantage) or None
        if the buffer is too small.
        """
        if len(self.memory) < cfg.AC_BATCH_SIZE // 4:
            return None

        batch = self.memory.sample(min(cfg.AC_BATCH_SIZE, len(self.memory)))

        # Pre-compute advantages for normalisation (standard PPO practice)
        raw_advs = []
        for ex in batch:
            _, v_s,  _ = self.forward(ex["s"])
            _, v_ns, _ = self.forward(ex["ns"])
            td = ex["r"] + (0.0 if ex["done"] else self.gamma * v_ns)
            raw_advs.append(td - v_s)
        adv_arr  = np.array(raw_advs, dtype=np.float32)
        adv_mean = float(adv_arr.mean())
        adv_std  = float(adv_arr.std()) + 1e-8

        total_actor_loss  = 0.0
        total_critic_loss = 0.0
        total_advantage   = 0.0

        for _ in range(cfg.AC_PPO_K_EPOCHS):
            for ex in batch:
                s, a, r, ns, done, old_lp = (
                    ex["s"], ex["a"], ex["r"], ex["ns"], ex["done"], ex["lp"]
                )

                probs, value, hidden = self.forward(s)
                _, next_value, _     = self.forward(ns)

                # TD target and normalised advantage
                target    = r + (0.0 if done else self.gamma * next_value)
                advantage = (target - value - adv_mean) / adv_std

                # ── PPO ratio (clamped to prevent exp overflow) ───────────
                new_lp = float(np.log(probs[a] + 1e-8))
                ratio  = float(np.clip(np.exp(new_lp - old_lp), 0.0, 10.0))

                # ── L_clip gradient w.r.t. logits (ascent direction) ──────
                is_clipped = (
                    ratio < 1 - cfg.AC_PPO_EPSILON
                    or ratio > 1 + cfg.AC_PPO_EPSILON
                )
                if is_clipped:
                    d_clip = np.zeros(self.as_, dtype=np.float32)
                else:
                    one_hot    = np.zeros(self.as_, dtype=np.float32)
                    one_hot[a] = 1.0
                    d_clip     = (ratio * advantage * (one_hot - probs)).astype(np.float32)

                # ── Entropy gradient: π_i * (-log(π_i) - H) ──────────────
                log_probs = np.log(probs + 1e-8)
                H         = -float(np.sum(probs * log_probs))
                d_entropy = (probs * (-log_probs - H)).astype(np.float32)

                # Combined actor gradient (ascent on L_clip + c2*H)
                d_actor = d_clip + cfg.AC_PPO_C2 * d_entropy

                # ── Actor update ───────────────────────────────────────────
                self.W2a += self.lr * np.outer(d_actor, hidden)
                self.b2a += self.lr * d_actor

                # ── Critic update (ascent on -c1 * advantage^2) ───────────
                self.W2c += self.lr * cfg.AC_PPO_C1 * 2.0 * advantage * hidden[None, :]
                self.b2c += self.lr * cfg.AC_PPO_C1 * 2.0 * advantage

                # ── Hidden layer update ────────────────────────────────────
                d_hidden  = self.W2a.T @ (self.lr * d_actor)
                d_hidden += self.lr * cfg.AC_PPO_C1 * 2.0 * advantage * self.W2c[0]
                d_hidden *= (hidden > 0).astype(np.float32)

                self.W1 += 0.008 * np.outer(d_hidden, s)
                self.b1 += 0.008 * d_hidden

                total_actor_loss  += float(np.log(probs[a] + 1e-8) * advantage)
                total_critic_loss += float(advantage ** 2)
                total_advantage   += float(advantage)

        n = len(batch)
        return dict(
            actor_loss  = -total_actor_loss  / n,
            critic_loss =  total_critic_loss / n,
            mean_adv    =  total_advantage   / n,
        )

    # ── reproduction ─────────────────────────────────────────────────────────

    def breed(self, other: "ActorCritic") -> "ActorCritic":
        """
        Create a child brain by crossing over weights from self and *other*,
        applying small Gaussian mutations, and seeding memory from both parents.

        Parameters
        ----------
        other : the second parent's ActorCritic

        Returns
        -------
        ActorCritic  child with inherited weights + knowledge transfer
        """
        child = ActorCritic(self.ss, self.as_, self.hs)

        def _crossover(A: np.ndarray, B: np.ndarray) -> np.ndarray:
            mask    = (np.random.rand(*A.shape) < 0.5).astype(np.float32)
            merged  = mask * A + (1.0 - mask) * B
            mutate  = (np.random.rand(*A.shape) < cfg.AC_MUTATION_RATE).astype(np.float32)
            noise   = np.random.randn(*A.shape).astype(np.float32) * cfg.AC_MUTATION_STD
            return (merged + noise * mutate).astype(np.float32)

        child.W1  = _crossover(self.W1,  other.W1)
        child.b1  = _crossover(self.b1,  other.b1)
        child.W2a = _crossover(self.W2a, other.W2a)
        child.b2a = _crossover(self.b2a, other.b2a)
        child.W2c = _crossover(self.W2c, other.W2c)
        child.b2c = _crossover(self.b2c, other.b2c)

        # Knowledge transfer: seed child replay buffer with parents' memories
        inherited = (
            self.memory.sample(cfg.AC_INHERIT_MEM)
            + other.memory.sample(cfg.AC_INHERIT_MEM)
        )
        child.memory.seed(inherited)

        return child

    # ── serialisation helpers ─────────────────────────────────────────────────

    def get_weights(self) -> dict:
        """Return all weights as a plain dict (for saving/logging)."""
        return dict(
            W1=self.W1, b1=self.b1,
            W2a=self.W2a, b2a=self.b2a,
            W2c=self.W2c, b2c=self.b2c,
        )

    def set_weights(self, weights: dict) -> None:
        """Restore weights from a dict produced by get_weights()."""
        self.W1  = weights["W1"].astype(np.float32)
        self.b1  = weights["b1"].astype(np.float32)
        self.W2a = weights["W2a"].astype(np.float32)
        self.b2a = weights["b2a"].astype(np.float32)
        self.W2c = weights["W2c"].astype(np.float32)
        self.b2c = weights["b2c"].astype(np.float32)
