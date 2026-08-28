"""Synthetic RF environment (spec section 6.1).

Each band is a hidden 2-state Markov chain (OFF/ON) with per-band transition
probabilities P01_i (OFF->ON) and P10_i (ON->OFF) that are hidden from the
scheduler. Three emitter archetypes sit on top of the base Markov behaviour:

1. ``markov``    - static/random Gilbert-Elliot band.
2. ``periodic``  - rotating-antenna / periodic-scan emitter: ON for a short
                   dwell once every ``period`` slots (+/- jitter). This is the
                   "periodic scan receiver/emitter" the problem statement calls
                   out explicitly.
3. ``hopper``    - a frequency-agile emitter that hops across a subset of bands
                   following a pseudo-random hop sequence; only the band it is
                   currently sitting on is ON. Satisfies "frequency-agile
                   emitters".

Defaults for periods / hop rates / duty cycles are seeded from statistics
extracted from the real TSRD sample (see backend/data/prepare_tsrd_cache.py and
data/tsrd_defaults.json) so that "Synthetic" mode looks like a realistic EW
environment even when the real cache is not loaded.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base_environment import BandInfo, RFEnvironment

_DEFAULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tsrd_defaults.json")


def _load_tsrd_defaults() -> dict:
    """Load TSRD-derived default ranges if available, else fall back to
    reasonable hand values. Never raises - the file is optional."""
    fallback = {
        "p01_range": [0.01, 0.08],
        "p10_range": [0.15, 0.45],
        "periodic_period_range": [18, 60],
        "periodic_dwell_range": [1, 3],
        "periodic_jitter": 1,
        "hopper_hopset_range": [3, 6],
        "hopper_dwell_range": [1, 2],
        "duty_cycle_range": [0.04, 0.20],
        "source": "builtin_fallback",
    }
    try:
        with open(os.path.abspath(_DEFAULTS_PATH), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        fallback.update({k: v for k, v in data.items() if v is not None})
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return fallback


@dataclass
class EmitterSpec:
    """Resolved per-band emitter configuration."""

    kind: str  # markov | periodic | hopper | quiet
    p01: float = 0.02
    p10: float = 0.3
    period: int = 30
    dwell: int = 2
    jitter: int = 1
    phase: int = 0
    hop_group: int = -1  # id of the hopper this band belongs to (-1 = none)


class SyntheticEnvironment(RFEnvironment):
    """Seedable synthetic multi-band RF environment."""

    def __init__(
        self,
        n_bands: int = 24,
        p_miss: float = 0.1,
        p_fa: float = 0.05,
        seed: int = 0,
        emitter_mix: Optional[dict] = None,
        defaults: Optional[dict] = None,
    ) -> None:
        super().__init__(n_bands=n_bands, p_miss=p_miss, p_fa=p_fa)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.defaults = defaults or _load_tsrd_defaults()
        # Fraction of bands per archetype. Remaining -> quiet.
        self.emitter_mix = emitter_mix or {
            "markov": 0.45,
            "periodic": 0.25,
            "hopper": 0.20,
            "quiet": 0.10,
        }
        self._build_bands()
        self.reset()

    # ------------------------------------------------------------ setup
    def _build_bands(self) -> None:
        d = self.defaults
        rng = self.rng
        n = self.n_bands

        kinds = ["markov", "periodic", "hopper", "quiet"]
        probs = np.array([self.emitter_mix.get(k, 0.0) for k in kinds], dtype=float)
        if probs.sum() <= 0:
            probs = np.array([1.0, 0.0, 0.0, 0.0])
        probs = probs / probs.sum()
        assigned = rng.choice(len(kinds), size=n, p=probs)

        self.specs: list[EmitterSpec] = []
        hop_groups: dict[int, list[int]] = {}
        next_hop_group = 0

        for i in range(n):
            kind = kinds[assigned[i]]
            if kind == "markov":
                p01 = float(rng.uniform(*d["p01_range"]))
                p10 = float(rng.uniform(*d["p10_range"]))
                self.specs.append(EmitterSpec(kind="markov", p01=p01, p10=p10))
            elif kind == "periodic":
                period = int(rng.integers(d["periodic_period_range"][0], d["periodic_period_range"][1] + 1))
                dwell = int(rng.integers(d["periodic_dwell_range"][0], d["periodic_dwell_range"][1] + 1))
                phase = int(rng.integers(0, period))
                self.specs.append(
                    EmitterSpec(
                        kind="periodic",
                        period=period,
                        dwell=dwell,
                        jitter=int(d["periodic_jitter"]),
                        phase=phase,
                    )
                )
            elif kind == "hopper":
                self.specs.append(EmitterSpec(kind="hopper", hop_group=next_hop_group))
                hop_groups.setdefault(next_hop_group, []).append(i)
                # Close a hop group after it reaches a random size.
                target = int(rng.integers(d["hopper_hopset_range"][0], d["hopper_hopset_range"][1] + 1))
                if len(hop_groups[next_hop_group]) >= target:
                    next_hop_group += 1
            else:  # quiet
                self.specs.append(EmitterSpec(kind="quiet", p01=0.0, p10=1.0))

        # Build hopper schedules: for each group, a pseudo-random band-visit
        # sequence and per-visit dwell.
        self.hop_groups = {gid: bands for gid, bands in hop_groups.items() if bands}
        self._hop_dwell = int(rng.integers(d["hopper_dwell_range"][0], d["hopper_dwell_range"][1] + 1))

    # ------------------------------------------------------------ lifecycle
    def reset(self) -> np.ndarray:
        self.t = 0
        # Reset RNG so a scenario replays identically for every strategy pass.
        self.rng = np.random.default_rng(self.seed)
        self._build_bands()  # rebuild with the fresh rng for full determinism
        self.state = np.zeros(self.n_bands, dtype=np.int8)

        # Initialise Markov bands at their stationary distribution.
        for i, spec in enumerate(self.specs):
            if spec.kind == "markov":
                denom = spec.p01 + spec.p10
                pi_on = spec.p01 / denom if denom > 0 else 0.0
                self.state[i] = 1 if self.rng.random() < pi_on else 0

        # Hopper group cursors.
        self._hop_pos = {gid: 0 for gid in self.hop_groups}
        self._hop_timer = {gid: 0 for gid in self.hop_groups}
        self._apply_hoppers()
        return self.state.copy()

    def _apply_hoppers(self) -> None:
        for gid, bands in self.hop_groups.items():
            for b in bands:
                self.state[b] = 0
            cur = bands[self._hop_pos[gid] % len(bands)]
            self.state[cur] = 1

    # ------------------------------------------------------------ dynamics
    def step(self) -> np.ndarray:
        self.t += 1
        rng = self.rng
        for i, spec in enumerate(self.specs):
            if spec.kind == "markov":
                if self.state[i] == 1:
                    self.state[i] = 0 if rng.random() < spec.p10 else 1
                else:
                    self.state[i] = 1 if rng.random() < spec.p01 else 0
            elif spec.kind == "periodic":
                jitter = int(rng.integers(-spec.jitter, spec.jitter + 1)) if spec.jitter > 0 else 0
                phase_in_period = (self.t + spec.phase) % spec.period
                on = 0 <= (phase_in_period + jitter) < spec.dwell
                self.state[i] = 1 if on else 0
            elif spec.kind == "quiet":
                self.state[i] = 0
            # hopper bands are set by _apply_hoppers below

        # Advance hoppers.
        for gid, bands in self.hop_groups.items():
            self._hop_timer[gid] += 1
            if self._hop_timer[gid] >= self._hop_dwell:
                self._hop_timer[gid] = 0
                # pseudo-random next index (avoid staying put)
                step = int(rng.integers(1, len(bands))) if len(bands) > 1 else 0
                self._hop_pos[gid] = (self._hop_pos[gid] + step) % len(bands)
        self._apply_hoppers()

        return self.state.copy()

    # ------------------------------------------------------------ accessors
    @property
    def ground_truth_state(self) -> np.ndarray:
        return self.state

    def band_info(self) -> list[BandInfo]:
        infos = []
        for i, spec in enumerate(self.specs):
            stats = {}
            if spec.kind == "markov":
                stats = {"p01": round(spec.p01, 4), "p10": round(spec.p10, 4)}
            elif spec.kind == "periodic":
                stats = {"period": spec.period, "dwell": spec.dwell}
            elif spec.kind == "hopper":
                stats = {"hop_group": spec.hop_group}
            infos.append(
                BandInfo(index=i, label=f"B{i:02d}", emitter_type=spec.kind, stats=stats)
            )
        return infos
