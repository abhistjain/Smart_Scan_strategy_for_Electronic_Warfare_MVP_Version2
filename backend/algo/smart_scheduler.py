"""Smart Scheduler - the ML-based ES receiver scheduler (the core deliverable).

Composes the four algorithmic pieces into one Top-M policy:

    Thompson sampling  ->  sampled (P01, P10) per band  (no prior intelligence)
            |
    Bayesian belief filter  ->  predicted belief b_i(t|t-1)
            |
    Whittle index engine    ->  W_i(t) for ALL bands
            |  + Lomb-Scargle "intercept-ahead" boost  + UCB exploration bonus
    Top-M selection         ->  the M bands actually scanned

Timeline per tick:
    select(t):   sample -> predict belief -> index all bands -> add boosts -> top-M
    observe_feedback(scanned, obs, t): Bayes update, Thompson update, periodicity
"""

from __future__ import annotations

import numpy as np

from .baselines import Strategy
from .belief_filter import BeliefFilter
from .periodicity import PeriodicityDetector
from .thompson import ThompsonTransitionLearner
from .whittle_index import WhittleIndexEngine


class SmartScheduler(Strategy):
    name = "smart"

    def __init__(
        self,
        n_bands: int,
        m: int,
        p_miss: float,
        p_fa: float,
        seed: int = 0,
        beta: float = 0.99,
        reward: float = 1.0,
        ucb_c: float = 0.05,
        use_periodicity: bool = True,
        use_thompson: bool = True,
    ) -> None:
        super().__init__(n_bands, m, seed)
        self.belief = BeliefFilter(n_bands, p_miss, p_fa, init_belief=0.5)
        self.thompson = ThompsonTransitionLearner(n_bands, seed=seed + 1)
        self.whittle = WhittleIndexEngine(n_bands, beta=beta, reward=reward)
        self.periodicity = PeriodicityDetector(n_bands)
        self.ucb_c = float(ucb_c)
        self.use_periodicity = bool(use_periodicity)
        self.use_thompson = bool(use_thompson)
        self._p01 = np.full(n_bands, 0.05)
        self._p10 = np.full(n_bands, 0.3)
        self._last_index = np.zeros(n_bands)

    def reset(self) -> None:
        self.belief.reset(0.5)
        self.thompson.reset()
        self.periodicity.reset()
        self._p01[:] = 0.05
        self._p10[:] = 0.3
        self._last_index[:] = 0.0

    def select(self, t: int) -> np.ndarray:
        # 1. Sample transition probabilities (exploration via posterior sampling).
        if self.use_thompson:
            self._p01, self._p10 = self.thompson.sample()
        # 2. Bayesian prediction step -> prior belief for this tick.
        belief_pred = self.belief.predict(self._p01, self._p10)
        # 3. Whittle index for every band.
        idx = self.whittle.index_array(belief_pred, self._p01, self._p10)
        # 4a. UCB exploration bonus (transition probs learned online).
        idx = self.whittle.add_exploration_bonus(
            idx, t, self.thompson.counts(), c=self.ucb_c
        )
        # 4b. Periodicity "intercept-ahead" boost.
        if self.use_periodicity:
            idx = idx + self.periodicity.index_boost(t)
        self._last_index = idx
        # 5. Top-M selection.
        top = np.argpartition(-idx, min(self.m, self.n_bands) - 1)[: self.m]
        # Order the chosen bands by index descending for stable presentation.
        top = top[np.argsort(-idx[top])]
        return top.astype(np.int64)

    def observe_feedback(self, scanned: np.ndarray, obs: np.ndarray, t: int) -> None:
        scanned = np.asarray(scanned, dtype=np.int64)
        obs = np.asarray(obs, dtype=np.int64)
        # Bayesian update on scanned bands.
        self.belief.update(scanned, obs)
        # Thompson posterior update from consecutive-scan transitions.
        if self.use_thompson:
            self.thompson.update(scanned, obs)
        # Periodicity buffers.
        if self.use_periodicity:
            for b, o in zip(scanned, obs):
                if o == 1:
                    self.periodicity.record_hit(int(b), t)
            self.periodicity.update(t)

    # ------------------------------------------------------------ introspection
    @property
    def beliefs(self) -> np.ndarray:
        return self.belief.belief

    @property
    def index(self) -> np.ndarray:
        return self._last_index

    def predicted_next_active(self) -> dict[int, float]:
        return self.periodicity.predicted_next_active()

    def periodicity_summary(self) -> list[dict]:
        return self.periodicity.summary()
