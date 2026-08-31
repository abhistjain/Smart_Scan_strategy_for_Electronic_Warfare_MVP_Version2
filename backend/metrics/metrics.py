"""Figures of merit, computed per-strategy and live (spec section 6.9).

All metrics are accumulated online against ground truth (available to the
scorer, never to the scheduler). Definitions (kept distinct per the problem
statement's wording):

- Probability of Detection (Pd)
      fraction of ALL truly-active (band, tick) cells that were intercepted
      (scanned AND observed o=1). This is the coverage-oriented number that shows
      the scheduler winning the 2-D search.
- Probability of False Alarm (Pfa)
      fraction of scans of truly-OFF bands that returned o=1.
- Sensitivity
      receiver-level detection rate = detections / scans of truly-ON bands
      (i.e. 1 - empirical miss rate). Distinct from Pd, which is over all active
      cells whether scanned or not.
- Average Intercept Rate
      successful intercepts per unit simulated time (per tick).
- Average Reward/Cost
      running mean of the section 6.8 reward.
- % Correct Predictions
      fraction of scanned bands whose pre-scan prediction (in Top-M / belief>0.5)
      matched the ground-truth state.
- Average Intercept Time Error
      mean absolute error between predicted and actual next-active tick for
      periodic/predictable emitters (smart scheduler only).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MetricsAccumulator:
    n_bands: int

    # Confusion-matrix-style counters vs ground truth.
    total_active_cells: int = 0          # denominator for Pd
    intercepted_active_cells: int = 0    # numerator for Pd

    scans_on_active: int = 0             # denominator for sensitivity
    detections_on_active: int = 0        # numerator for sensitivity

    scans_on_idle: int = 0               # denominator for Pfa
    false_alarms: int = 0                # numerator for Pfa

    correct_predictions: int = 0
    total_predictions: int = 0

    total_intercepts: int = 0
    ticks: int = 0

    reward_sum: float = 0.0

    # Intercept time error (predicted vs actual next active onset).
    time_error_sum: float = 0.0
    time_error_count: int = 0

    # Rolling history for charts.
    cumulative_intercepts: list[int] = field(default_factory=list)
    intercept_rate_history: list[float] = field(default_factory=list)
    reward_history: list[float] = field(default_factory=list)

    def update(
        self,
        scanned_bands: np.ndarray,
        observations: np.ndarray,
        ground_truth: np.ndarray,
        reward: float,
        predicted_active_bands: np.ndarray | None = None,
    ) -> None:
        scanned_bands = np.asarray(scanned_bands, dtype=np.int64)
        observations = np.asarray(observations, dtype=np.int64)
        ground_truth = np.asarray(ground_truth, dtype=np.int64)

        self.ticks += 1

        active_mask = ground_truth == 1
        self.total_active_cells += int(active_mask.sum())

        scanned_mask = np.zeros(self.n_bands, dtype=bool)
        scanned_mask[scanned_bands] = True

        truth_scanned = ground_truth[scanned_bands]

        # Sensitivity + Pd numerators: detections on truly-active scanned bands.
        det_active = int(np.sum((truth_scanned == 1) & (observations == 1)))
        self.detections_on_active += det_active
        self.scans_on_active += int(np.sum(truth_scanned == 1))
        self.intercepted_active_cells += det_active

        # Pfa: false alarms on truly-idle scanned bands.
        self.scans_on_idle += int(np.sum(truth_scanned == 0))
        self.false_alarms += int(np.sum((truth_scanned == 0) & (observations == 1)))

        # Intercept counting.
        self.total_intercepts += det_active
        self.reward_sum += float(reward)

        # % correct predictions: predicted-active bands (Top-M) vs truth.
        if predicted_active_bands is not None and len(predicted_active_bands) > 0:
            pred = np.asarray(predicted_active_bands, dtype=np.int64)
            self.total_predictions += pred.size
            self.correct_predictions += int(np.sum(ground_truth[pred] == 1))

        # History.
        self.cumulative_intercepts.append(self.total_intercepts)
        self.intercept_rate_history.append(self.total_intercepts / max(self.ticks, 1))
        self.reward_history.append(self.reward_sum / max(self.ticks, 1))

    def record_time_error(self, abs_error: float) -> None:
        self.time_error_sum += float(abs_error)
        self.time_error_count += 1

    # ------------------------------------------------------------ readouts
    def snapshot(self) -> dict:
        pd = self.intercepted_active_cells / self.total_active_cells if self.total_active_cells else 0.0
        pfa = self.false_alarms / self.scans_on_idle if self.scans_on_idle else 0.0
        sens = self.detections_on_active / self.scans_on_active if self.scans_on_active else 0.0
        pct = self.correct_predictions / self.total_predictions if self.total_predictions else 0.0
        rate = self.total_intercepts / self.ticks if self.ticks else 0.0
        avg_reward = self.reward_sum / self.ticks if self.ticks else 0.0
        time_err = self.time_error_sum / self.time_error_count if self.time_error_count else None
        return {
            "pd": round(pd, 4),
            "pfa": round(pfa, 4),
            "sensitivity": round(sens, 4),
            "intercept_rate": round(rate, 4),
            "avg_reward": round(avg_reward, 4),
            "pct_correct": round(pct, 4),
            "time_error": round(time_err, 4) if time_err is not None else None,
            "total_intercepts": self.total_intercepts,
            "ticks": self.ticks,
        }
