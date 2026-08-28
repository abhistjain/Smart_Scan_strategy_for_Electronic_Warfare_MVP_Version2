"""Shared RFEnvironment interface.

Both the synthetic generator and the TSRD replay environment implement this
interface, so the scheduler / metrics / API layers never need to know which
data source they are talking to (spec section 6.1 / 6.2).

Design notes on fairness (important for the strategy comparison):

* The hidden ground-truth trajectory is advanced by ``step()`` and is driven
  entirely by the environment's own seeded RNG. It does NOT depend on which
  bands are scanned, so a single environment instance can serve all four
  strategies simultaneously on a byte-identical trajectory.
* Sensor noise, however, must be *independent* between strategies (otherwise
  one strategy's "lucky" false-alarm draw would be forced onto the others).
  ``observe()`` therefore takes the caller's own RNG.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class BandInfo:
    """Static metadata about one frequency band, used by the UI side panel."""

    index: int
    label: str
    # "markov" | "periodic" | "hopper" | "quiet" | "tsrd"  (tsrd = inferred from real data)
    emitter_type: str
    # Optional descriptive stats (period in slots, hop-set size, duty cycle...)
    stats: dict = field(default_factory=dict)


class RFEnvironment(ABC):
    """Abstract RF environment: hidden per-band ON/OFF truth + noisy scans."""

    def __init__(self, n_bands: int, p_miss: float, p_fa: float) -> None:
        self.n_bands = int(n_bands)
        self.p_miss = float(p_miss)
        self.p_fa = float(p_fa)
        self.t = 0

    # ------------------------------------------------------------------ core
    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset to t=0 and return the initial ground-truth state vector."""

    @abstractmethod
    def step(self) -> np.ndarray:
        """Advance the hidden truth by one time slot; return the new state
        vector (dtype=int8, 1 = transmitting, 0 = silent)."""

    @property
    @abstractmethod
    def ground_truth_state(self) -> np.ndarray:
        """Current hidden state vector. Only the metrics engine and the
        'Instructor Mode' overlay may look at this - never the scheduler."""

    # ------------------------------------------------------------ observation
    def observe(self, bands: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Noisy observation model applied to the scanned bands only.

        P(o=1 | state=1) = 1 - p_miss
        P(o=1 | state=0) = p_fa

        Unscanned bands yield no observation at all (partial observability).
        ``rng`` belongs to the calling strategy so noise draws are independent
        across strategies while the underlying truth stays identical.
        """
        bands = np.asarray(bands, dtype=np.int64)
        truth = self.ground_truth_state[bands]
        u = rng.random(bands.shape[0])
        obs = np.where(truth == 1, (u > self.p_miss), (u < self.p_fa))
        return obs.astype(np.int8)

    # ---------------------------------------------------------------- meta
    @abstractmethod
    def band_info(self) -> list[BandInfo]:
        """Static per-band metadata for the UI."""

    @property
    def duration(self) -> Optional[int]:
        """Total number of slots if finite (replay), else None (endless)."""
        return None
