"""Thompson sampling for unknown transition probabilities (spec section 6.4).

The problem statement forbids assuming prior reliable intelligence, so we do NOT
know each band's Markov transition probabilities. We place Beta(1,1) priors on

    P01_i = P(OFF -> ON)   and   P10_i = P(ON -> OFF)

and update them from observed *consecutive-scan transitions*: when a band is
scanned on two adjacent timesteps we get a (prev_obs -> curr_obs) pair that acts
as a noisy sample of the underlying state transition.

Caveat documented for honesty: observations are noisy proxies for the true
state (p_miss / p_fa), so these counts are approximate. In practice this still
concentrates the posteriors correctly because the noise is unbiased around the
true transition, and it keeps the scheduler fully "no prior intelligence".

At each decision epoch we SAMPLE from the posteriors (not take the mean) - this
sampling *is* the exploration mechanism, replacing any ad-hoc epsilon-greedy.
"""

from __future__ import annotations

import numpy as np


class ThompsonTransitionLearner:
    def __init__(self, n_bands: int, seed: int = 0) -> None:
        self.n_bands = int(n_bands)
        self.rng = np.random.default_rng(seed)
        # Beta(alpha, beta) for each transition kind, initialised uniform (1,1).
        self.a01 = np.ones(self.n_bands)  # OFF->ON  successes (became ON)
        self.b01 = np.ones(self.n_bands)  # OFF->OFF (stayed OFF)
        self.a10 = np.ones(self.n_bands)  # ON->OFF  successes (became OFF)
        self.b10 = np.ones(self.n_bands)  # ON->ON  (stayed ON)
        # Track last observation per band + whether it was scanned last tick.
        self._last_obs = np.full(self.n_bands, -1, dtype=np.int64)

    def reset(self) -> None:
        self.a01[:] = 1.0
        self.b01[:] = 1.0
        self.a10[:] = 1.0
        self.b10[:] = 1.0
        self._last_obs[:] = -1

    def update(self, scanned_bands: np.ndarray, observations: np.ndarray) -> None:
        """Update posteriors from consecutive-scan transitions, then remember
        this tick's observations for the next update."""
        scanned_bands = np.asarray(scanned_bands, dtype=np.int64)
        observations = np.asarray(observations, dtype=np.int64)

        new_last = np.full(self.n_bands, -1, dtype=np.int64)
        for band, obs in zip(scanned_bands, observations):
            prev = self._last_obs[band]
            if prev == 0:  # was OFF -> observe transition to obs
                if obs == 1:
                    self.a01[band] += 1.0  # OFF->ON
                else:
                    self.b01[band] += 1.0  # OFF->OFF
            elif prev == 1:  # was ON
                if obs == 0:
                    self.a10[band] += 1.0  # ON->OFF
                else:
                    self.b10[band] += 1.0  # ON->ON
            new_last[band] = obs
        # Bands not scanned this tick lose their "last" (transition must be from
        # two *consecutive* scans to be a valid one-step transition).
        self._last_obs = new_last

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        """Draw one posterior sample of (P01, P10) for every band."""
        p01 = self.rng.beta(self.a01, self.b01)
        p10 = self.rng.beta(self.a10, self.b10)
        return p01, p10

    def mean(self) -> tuple[np.ndarray, np.ndarray]:
        """Posterior means (useful for logging / the belief prediction fallback)."""
        p01 = self.a01 / (self.a01 + self.b01)
        p10 = self.a10 / (self.a10 + self.b10)
        return p01, p10

    def counts(self) -> np.ndarray:
        """Number of transitions observed per band (for UCB exploration bonus)."""
        return (self.a01 + self.b01 - 2.0) + (self.a10 + self.b10 - 2.0)
