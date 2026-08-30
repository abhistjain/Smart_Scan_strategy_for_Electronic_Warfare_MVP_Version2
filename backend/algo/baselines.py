"""Baseline scanning strategies (spec section 6.7).

All strategies share the same interface: given the current tick and their own
observation RNG, return the set of M bands to scan. They run on the IDENTICAL
environment trajectory as the smart scheduler for a fair comparison.

1. SequentialSweep  - fixed round-robin order, no adaptation. This is the literal
                      open-loop incumbent the problem statement asks us to beat.
2. RandomStrategy   - uniformly random M bands per tick.
3. GreedyRecentHit  - always re-scan the M most-recently-hit bands (fall back to
                      round-robin to fill spare capacity), no real exploration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Strategy(ABC):
    name: str = "base"

    def __init__(self, n_bands: int, m: int, seed: int = 0) -> None:
        self.n_bands = int(n_bands)
        self.m = int(m)
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:  # pragma: no cover - trivial
        pass

    @abstractmethod
    def select(self, t: int) -> np.ndarray:
        """Return an int array of M band indices to scan at tick t."""

    def observe_feedback(self, scanned: np.ndarray, obs: np.ndarray, t: int) -> None:
        """Optional hook to receive this tick's observations."""


class SequentialSweep(Strategy):
    name = "sequential"

    def __init__(self, n_bands: int, m: int, seed: int = 0) -> None:
        super().__init__(n_bands, m, seed)
        self._cursor = 0

    def reset(self) -> None:
        self._cursor = 0

    def select(self, t: int) -> np.ndarray:
        idx = (self._cursor + np.arange(self.m)) % self.n_bands
        self._cursor = (self._cursor + self.m) % self.n_bands
        return idx.astype(np.int64)


class RandomStrategy(Strategy):
    name = "random"

    def select(self, t: int) -> np.ndarray:
        return self.rng.choice(self.n_bands, size=self.m, replace=False).astype(np.int64)


class GreedyRecentHit(Strategy):
    name = "greedy"

    def __init__(self, n_bands: int, m: int, seed: int = 0) -> None:
        super().__init__(n_bands, m, seed)
        self._last_hit_time = np.full(n_bands, -1, dtype=np.int64)
        self._cursor = 0

    def reset(self) -> None:
        self._last_hit_time[:] = -1
        self._cursor = 0

    def select(self, t: int) -> np.ndarray:
        # Rank bands by most-recent hit time (descending). Bands never hit sort
        # last; fill any remaining slots via a round-robin sweep.
        order = np.argsort(-self._last_hit_time)
        hit_bands = order[self._last_hit_time[order] >= 0][: self.m]
        chosen = list(hit_bands)
        while len(chosen) < self.m:
            cand = self._cursor % self.n_bands
            self._cursor += 1
            if cand not in chosen:
                chosen.append(cand)
        return np.array(chosen[: self.m], dtype=np.int64)

    def observe_feedback(self, scanned: np.ndarray, obs: np.ndarray, t: int) -> None:
        scanned = np.asarray(scanned, dtype=np.int64)
        obs = np.asarray(obs, dtype=np.int64)
        for b, o in zip(scanned, obs):
            if o == 1:
                self._last_hit_time[b] = t
