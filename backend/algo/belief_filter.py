"""Bayesian recursive belief filter (spec section 6.3).

For each band i we track b_i(t) = P(state_i(t) = ON | observations up to t),
using a 2-state HMM with transition probabilities (P01, P10) that may be
supplied per-timestep by Thompson sampling (so they can be learned online).

Prediction (every timestep, every band):
    b_pred = b*(1 - P10) + (1 - b)*P01

Update (only for scanned bands, exact Bayes with explicit normalisation):
    Let e1 = P(o | ON), e0 = P(o | OFF) under the observation model
        P(o=1|ON) = 1 - p_miss ,  P(o=0|ON) = p_miss
        P(o=1|OFF) = p_fa      ,  P(o=0|OFF) = 1 - p_fa
    num_on  = e1 * b_pred
    num_off = e0 * (1 - b_pred)
    b_post  = num_on / (num_on + num_off)          # denominator written out

Unscanned bands keep b_i(t) = b_pred (evolve by prediction only).
"""

from __future__ import annotations

import numpy as np


class BeliefFilter:
    def __init__(
        self,
        n_bands: int,
        p_miss: float,
        p_fa: float,
        init_belief: float = 0.5,
    ) -> None:
        self.n_bands = int(n_bands)
        self.p_miss = float(p_miss)
        self.p_fa = float(p_fa)
        self.belief = np.full(self.n_bands, float(init_belief), dtype=np.float64)

    def reset(self, init_belief: float = 0.5) -> None:
        self.belief[:] = init_belief

    def predict(self, p01: np.ndarray, p10: np.ndarray) -> np.ndarray:
        """Apply the Markov prediction step to every band. Returns predicted
        belief (also stored as the current belief until update() runs)."""
        p01 = np.asarray(p01, dtype=np.float64)
        p10 = np.asarray(p10, dtype=np.float64)
        self.belief = self.belief * (1.0 - p10) + (1.0 - self.belief) * p01
        np.clip(self.belief, 1e-9, 1.0 - 1e-9, out=self.belief)
        return self.belief

    def update(self, scanned_bands: np.ndarray, observations: np.ndarray) -> None:
        """Exact Bayes update for the scanned bands only. ``observations`` are
        aligned with ``scanned_bands`` (0/1)."""
        scanned_bands = np.asarray(scanned_bands, dtype=np.int64)
        observations = np.asarray(observations, dtype=np.int64)
        if scanned_bands.size == 0:
            return

        b_pred = self.belief[scanned_bands]

        # Likelihoods of the actual observation under each hidden state.
        # e1 = P(o | ON), e0 = P(o | OFF)
        e1 = np.where(observations == 1, 1.0 - self.p_miss, self.p_miss)
        e0 = np.where(observations == 1, self.p_fa, 1.0 - self.p_fa)

        num_on = e1 * b_pred
        num_off = e0 * (1.0 - b_pred)
        denom = num_on + num_off  # explicit normaliser; never approximated
        denom = np.where(denom <= 0, 1.0, denom)
        b_post = num_on / denom

        self.belief[scanned_bands] = np.clip(b_post, 1e-9, 1.0 - 1e-9)

    def step(
        self,
        p01: np.ndarray,
        p10: np.ndarray,
        scanned_bands: np.ndarray,
        observations: np.ndarray,
    ) -> np.ndarray:
        """Convenience: predict then update, returning the posterior belief."""
        self.predict(p01, p10)
        self.update(scanned_bands, observations)
        return self.belief.copy()
