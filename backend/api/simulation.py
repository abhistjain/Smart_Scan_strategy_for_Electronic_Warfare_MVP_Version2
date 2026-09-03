"""Simulation orchestrator (spec sections 5, 6.7).

Owns ONE environment instance and runs all four strategies on the byte-identical
ground-truth trajectory produced by that environment. Fairness guarantees:

* The hidden truth advances via env.step() and does not depend on any strategy's
  scans, so every strategy sees the same world at each tick.
* Each strategy has its OWN observation RNG, so sensor noise (p_miss / p_fa) is
  independent per strategy - one strategy's lucky false-alarm draw is not forced
  onto the others.

Produces the per-tick WebSocket payload described in spec section 7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from algo.baselines import GreedyRecentHit, RandomStrategy, SequentialSweep
from algo.reward import RewardWeights, compute_reward
from algo.smart_scheduler import SmartScheduler
from classification.engine import ClassificationEngine
from classification.priority_score import PriorityWeights
from metrics.metrics import MetricsAccumulator
from sim.base_environment import RFEnvironment
from sim.synthetic_environment import SyntheticEnvironment
from sim.tsrd_environment import TSRDEnvironment


@dataclass
class ScenarioConfig:
    data_source: str = "synthetic"  # "synthetic" | "real_tsrd"
    n_bands: int = 24
    m: int = 3
    seed: int = 0
    p_miss: float = 0.1
    p_fa: float = 0.05
    emitter_mix: Optional[dict] = None
    sample_id: int = 0  # for real_tsrd
    r_hit: float = 1.0
    c_dwell: float = 0.05
    c_miss_penalty: float = 0.5
    beta: float = 0.99
    ucb_c: float = 0.05
    # Threat Priority Score weights (v3 add-on Section 2.4). Sort-only.
    w_belief: float = 0.5
    w_conf: float = 0.2
    w_urgency: float = 0.3

    def to_dict(self) -> dict:
        return {
            "data_source": self.data_source,
            "n_bands": self.n_bands,
            "m": self.m,
            "seed": self.seed,
            "p_miss": self.p_miss,
            "p_fa": self.p_fa,
            "emitter_mix": self.emitter_mix,
            "sample_id": self.sample_id,
            "r_hit": self.r_hit,
            "c_dwell": self.c_dwell,
            "c_miss_penalty": self.c_miss_penalty,
            "beta": self.beta,
            "ucb_c": self.ucb_c,
            "w_belief": self.w_belief,
            "w_conf": self.w_conf,
            "w_urgency": self.w_urgency,
        }


def build_environment(cfg: ScenarioConfig) -> RFEnvironment:
    if cfg.data_source == "real_tsrd":
        return TSRDEnvironment(
            sample_id=cfg.sample_id, p_miss=cfg.p_miss, p_fa=cfg.p_fa, loop=True
        )
    return SyntheticEnvironment(
        n_bands=cfg.n_bands,
        p_miss=cfg.p_miss,
        p_fa=cfg.p_fa,
        seed=cfg.seed,
        emitter_mix=cfg.emitter_mix,
    )


class Simulation:
    STRATEGY_KEYS = ("smart", "sequential", "random", "greedy")

    def __init__(self, cfg: ScenarioConfig) -> None:
        self.cfg = cfg
        self.env = build_environment(cfg)
        self.n_bands = self.env.n_bands
        self.m = min(cfg.m, self.n_bands)
        self.weights = RewardWeights(cfg.r_hit, cfg.c_dwell, cfg.c_miss_penalty)

        # Strategies. Distinct seeds so each has an independent observation RNG.
        self.strategies = {
            "smart": SmartScheduler(
                self.n_bands, self.m, cfg.p_miss, cfg.p_fa,
                seed=cfg.seed, beta=cfg.beta, reward=cfg.r_hit, ucb_c=cfg.ucb_c,
            ),
            "sequential": SequentialSweep(self.n_bands, self.m, seed=cfg.seed),
            "random": RandomStrategy(self.n_bands, self.m, seed=cfg.seed + 100),
            "greedy": GreedyRecentHit(self.n_bands, self.m, seed=cfg.seed + 200),
        }
        # Independent observation RNGs per strategy.
        self.obs_rng = {
            k: np.random.default_rng(cfg.seed + 1000 + i)
            for i, k in enumerate(self.STRATEGY_KEYS)
        }
        self.metrics = {k: MetricsAccumulator(self.n_bands) for k in self.STRATEGY_KEYS}

        # Emitter behaviour classification + priority (v3 add-on). Sensing/
        # presentation only - no weapon/engagement logic (Section 0).
        self.classifier = ClassificationEngine(
            self.n_bands,
            weights=PriorityWeights(cfg.w_belief, cfg.w_conf, cfg.w_urgency),
        )
        self.classifier.set_band_hints(self.band_info())

        self.t = 0
        self._prev_truth = self.env.ground_truth_state.copy()
        # For smart intercept-time-error: predicted next active tick per band.
        self._smart_predictions: dict[int, float] = {}
        self.history: list[dict] = []  # compact per-tick record for export

    # ------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        self.env.reset()
        for s in self.strategies.values():
            s.reset()
        for k in self.STRATEGY_KEYS:
            self.metrics[k] = MetricsAccumulator(self.n_bands)
            self.obs_rng[k] = np.random.default_rng(self.cfg.seed + 1000 + self.STRATEGY_KEYS.index(k))
        self.classifier.reset()
        self.t = 0
        self._prev_truth = self.env.ground_truth_state.copy()
        self._smart_predictions = {}
        self.history = []

    # ------------------------------------------------------------ core tick
    def step(self) -> dict:
        # Advance the hidden truth once; all strategies see this same state.
        truth = self.env.step()
        self.t += 1

        # Detect OFF->ON onsets for intercept-time-error scoring (smart only).
        onsets = np.where((self._prev_truth == 0) & (truth == 1))[0]
        for b in onsets:
            pred = self._smart_predictions.get(int(b))
            if pred is not None:
                self.metrics["smart"].record_time_error(abs(pred - self.t))

        strat_payload: dict[str, dict] = {}
        smart_scanned = np.array([], dtype=np.int64)
        smart_obs = np.array([], dtype=np.int64)
        for key, strat in self.strategies.items():
            scanned = strat.select(self.t)
            obs = self.env.observe(scanned, self.obs_rng[key])
            if key == "smart":
                smart_scanned, smart_obs = scanned, obs
            reward = compute_reward(scanned, obs, truth, self.weights)
            # Predicted-active bands for %-correct = the scanned Top-M set.
            self.metrics[key].update(
                scanned_bands=scanned,
                observations=obs,
                ground_truth=truth,
                reward=reward,
                predicted_active_bands=scanned,
            )
            strat.observe_feedback(scanned, obs, self.t)

            entry = {
                "scanned_bands": scanned.tolist(),
                "hits": [int(b) for b, o in zip(scanned, obs) if o == 1],
                "metrics": self.metrics[key].snapshot(),
            }
            if key == "smart":
                entry["beliefs"] = [round(float(x), 4) for x in strat.beliefs]
                entry["index"] = [round(float(x), 4) for x in strat.index]
            strat_payload[key] = entry

        # Refresh smart predictions for the next tick's error scoring.
        self._smart_predictions = self.strategies["smart"].predicted_next_active()

        # Update emitter behaviour classification + priority from what the smart
        # receiver actually observed this tick (operator-available data only).
        smart_beliefs = self.strategies["smart"].beliefs
        periodicity = self.strategies["smart"].periodicity_summary()
        self.classifier.update(
            self.t, smart_scanned, smart_obs, smart_beliefs,
            periodicity, self._smart_predictions,
        )

        payload = {
            "t": self.t,
            "data_source": self.cfg.data_source,
            "strategies": strat_payload,
            "ground_truth": truth.astype(int).tolist(),
            "periodicity": periodicity,
            "classification": self.classifier.snapshot(),
        }
        self._prev_truth = truth.copy()

        # Keep a compact history row for export (metrics only, to bound memory).
        self.history.append(
            {"t": self.t, **{k: strat_payload[k]["metrics"] for k in self.STRATEGY_KEYS}}
        )
        return payload

    # ------------------------------------------------------------ helpers
    def band_info(self) -> list[dict]:
        return [
            {
                "index": bi.index,
                "label": bi.label,
                "emitter_type": bi.emitter_type,
                "stats": bi.stats,
            }
            for bi in self.env.band_info()
        ]

    def metrics_summary(self) -> dict:
        return {k: self.metrics[k].snapshot() for k in self.STRATEGY_KEYS}

    # --------------------------------------------------- classification helpers
    def band_detail(self, band: int) -> dict:
        detail = self.classifier.band_detail(band)
        info = self.env.band_info()[band]
        detail["label"] = info.label
        detail["emitter_type"] = info.emitter_type
        return detail

    def top_bands(self, n: int = 6) -> list[dict]:
        bands = self.classifier.top_bands(n)
        infos = self.env.band_info()
        for d in bands:
            b = d["band"]
            d["label"] = infos[b].label
        return bands

    def set_priority_weights(self, w_belief: float, w_conf: float, w_urgency: float) -> None:
        self.cfg.w_belief = w_belief
        self.cfg.w_conf = w_conf
        self.cfg.w_urgency = w_urgency
        self.classifier.set_weights(PriorityWeights(w_belief, w_conf, w_urgency))
