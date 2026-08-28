"""Reward / cost function (spec section 6.8).

reward(t) = sum over scanned bands i of:
    + R_hit           if o_i(t) = 1        (a successful intercept)
    - C_dwell         per scan             (opportunity cost of attention)
    - C_miss_penalty  if a truly-active band was NOT scanned this tick
                      (ground-truth-only term, used for scoring/comparison)

R_hit, C_dwell, C_miss_penalty are configurable and surfaced in the UI settings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RewardWeights:
    r_hit: float = 1.0
    c_dwell: float = 0.05
    c_miss_penalty: float = 0.5


def compute_reward(
    scanned_bands: np.ndarray,
    observations: np.ndarray,
    ground_truth: np.ndarray,
    weights: RewardWeights,
) -> float:
    """Compute the scalar reward for one timestep.

    ``ground_truth`` is the full per-band truth vector (used only for the miss
    penalty term, exactly as the spec specifies it is ground-truth-only scoring).
    """
    scanned_bands = np.asarray(scanned_bands, dtype=np.int64)
    observations = np.asarray(observations, dtype=np.int64)
    ground_truth = np.asarray(ground_truth, dtype=np.int64)

    hits = float(np.sum(observations == 1))
    dwell = float(scanned_bands.size)

    scanned_mask = np.zeros(ground_truth.shape[0], dtype=bool)
    scanned_mask[scanned_bands] = True
    missed_active = float(np.sum((ground_truth == 1) & (~scanned_mask)))

    return (
        weights.r_hit * hits
        - weights.c_dwell * dwell
        - weights.c_miss_penalty * missed_active
    )
