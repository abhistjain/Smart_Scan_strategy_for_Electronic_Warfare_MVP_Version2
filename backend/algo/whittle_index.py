"""Whittle Index engine for the 2-state restless bandit (spec section 6.5).

Closed-form Whittle index from:

    K. Liu and Q. Zhao, "Indexability of Restless Bandit Problems and
    Optimality of Whittle Index for Dynamic Multichannel Access," IEEE Trans.
    Information Theory, 56(11):5547-5567, 2010. (Theorem 2, discounted criterion)

We use the discounted form W_beta(omega) with beta -> 1 (default 0.99), which is
faithful to the paper, strictly increasing in the belief (nice for a stable
ranking), and avoids the constant-plateau subtlety of the average-reward form.

Notation (per band, all scalars):
    omega   current belief P(state = ON)
    p11 = 1 - p10  (ON -> ON)
    p01            (OFF -> ON)
    T(w)  = w*p11 + (1-w)*p01              one-step passive belief propagation
    w_o   = p01 / (1 - p11 + p01)          stationary belief (fixed point of T)
    L(x,w)= min{k : T^k(x) > w}            crossing time
    B     hit-reward weight (scales the index; ranking-invariant)

Positively correlated (p11 >= p01) and negatively correlated (p11 < p01) cases
are both implemented (Theorem 2, eq. 35 and 36).

A UCB-style exploration bonus  + c*sqrt(log t / n_i)  is added on top by
add_exploration_bonus(): a practical addition to pure Whittle theory because the
transition probabilities here are LEARNED online (Thompson) rather than known.
"""

from __future__ import annotations

import numpy as np

_L_CAP = 5000  # safety cap on the crossing-time iteration


def _T(w: float, p01: float, p11: float) -> float:
    return w * p11 + (1.0 - w) * p01


def _T_pow(x0: float, k: int, p01: float, p11: float) -> float:
    x = x0
    for _ in range(k):
        x = _T(x, p01, p11)
    return x


def _crossing_time(x0: float, w: float, p01: float, p11: float) -> int:
    """L(x0, w) = min{k >= 0 : T^k(x0) > w}, capped at _L_CAP."""
    if x0 > w:
        return 0
    x = x0
    for k in range(1, _L_CAP + 1):
        x = _T(x, p01, p11)
        if x > w:
            return k
    return _L_CAP


def whittle_index_scalar(
    omega: float, p01: float, p10: float, beta: float = 0.99, reward: float = 1.0
) -> float:
    """Closed-form Whittle index for a single band."""
    p01 = float(np.clip(p01, 1e-6, 1.0 - 1e-6))
    p10 = float(np.clip(p10, 1e-6, 1.0 - 1e-6))
    p11 = 1.0 - p10
    w = float(np.clip(omega, 0.0, 1.0))
    w_o = p01 / (1.0 - p11 + p01)

    if abs(p11 - p01) < 1e-9:
        # Independent channel: belief is memoryless; myopic index W = w.
        return w * reward

    if p11 >= p01:
        # ---- Case 1: positively correlated ------------------------------
        if w <= p01 or w >= p11:
            val = w
        elif w >= w_o:  # w_o <= w < p11
            val = w / (1.0 - beta * p11 + beta * w)
        else:  # p01 < w < w_o
            L = _crossing_time(p01, w, p01, p11)
            TL = _T_pow(p01, L, p01, p11)
            denom_c = (1.0 - beta * p11) * (1.0 - beta ** (L + 1)) + (
                1.0 - beta
            ) * beta ** (L + 1) * TL
            C1 = (1.0 - beta * p11) * (1.0 - beta ** L) / denom_c
            C2 = (beta ** L) * TL / denom_c
            a = w - beta * _T(w, p01, p11)  # (w - beta*T(w))
            g = beta * (1.0 - beta * p11) - beta * a
            num = a + C2 * (1.0 - beta) * g
            den = (1.0 - beta * p11) - C1 * g
            val = num / den if abs(den) > 1e-12 else w
    else:
        # ---- Case 2: negatively correlated (p11 < p01) ------------------
        Tp11 = _T(p11, p01, p11)
        base = 1.0 + (1.0 + beta) * beta * p01 - (beta ** 2) * Tp11
        C3 = (1.0 - beta * (1.0 - p01)) / base
        C4 = (beta * Tp11 * (1.0 - beta) + (beta ** 2) * p01) / base
        if w <= p11 or w >= p01:
            val = w
        elif w >= Tp11:  # T(p11) <= w < p01
            val = (beta * p01 + w * (1.0 - beta)) / (1.0 + beta * (p01 - w))
        elif w >= w_o:  # w_o <= w < T(p11)
            num = (1.0 - beta + beta * C4) * (beta * p01 + w * (1.0 - beta))
            den = 1.0 - beta * (1.0 - p01) - C3 * (
                (beta ** 2) * p01 + beta * w - (beta ** 2) * w
            )
            val = num / den if abs(den) > 1e-12 else w
        else:  # p11 < w < w_o
            Tw = _T(w, p01, p11)
            q = beta * Tw - beta * p01 - w
            num = (1.0 - beta) * (beta * p01 + w - beta * Tw) - C4 * beta * q
            den = 1.0 - beta * (1.0 - p01) + C3 * beta * q
            val = num / den if abs(den) > 1e-12 else w

    return float(val) * reward


class WhittleIndexEngine:
    """Vectorised per-band Whittle index computation for every timestep."""

    def __init__(self, n_bands: int, beta: float = 0.99, reward: float = 1.0) -> None:
        self.n_bands = int(n_bands)
        self.beta = float(beta)
        self.reward = float(reward)

    def index_array(
        self, belief: np.ndarray, p01: np.ndarray, p10: np.ndarray
    ) -> np.ndarray:
        """Compute W_i(t) for ALL bands (including idle ones, via their belief).

        Computing the index for every band each tick - not just recently hit
        ones - is the key advantage over the greedy baseline.
        """
        belief = np.asarray(belief, dtype=np.float64)
        p01 = np.asarray(p01, dtype=np.float64)
        p10 = np.asarray(p10, dtype=np.float64)
        out = np.empty(self.n_bands, dtype=np.float64)
        for i in range(self.n_bands):
            out[i] = whittle_index_scalar(
                belief[i], p01[i], p10[i], beta=self.beta, reward=self.reward
            )
        return out

    def add_exploration_bonus(
        self, index: np.ndarray, t: int, counts: np.ndarray, c: float = 0.05
    ) -> np.ndarray:
        """UCB-style bonus for bands with few observed transitions (learned
        online -> encourage sampling under-explored bands)."""
        t = max(t, 1)
        counts = np.asarray(counts, dtype=np.float64)
        bonus = c * np.sqrt(np.log(t + 1.0) / (counts + 1.0))
        return index + bonus
