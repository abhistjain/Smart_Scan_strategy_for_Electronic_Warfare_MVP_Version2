"""Classification engine: features -> label -> priority, wired per tick.

Keeps classification labels STABLE (Section 6): features are rolling averages and
labels are only re-evaluated every `classify_every` ticks with light hysteresis,
so chips don't flicker tick-to-tick. Priority (which depends on the fast-moving
belief) is recomputed every tick for responsive sorting.

Still strictly a sensing/presentation layer - no weapon/engagement logic
(add-on Section 0).
"""

from __future__ import annotations

import numpy as np

from .classifier import Classification, LABEL_UNCLASSIFIED, classify
from .features import FeatureExtractor
from .priority_score import PriorityWeights, compute_priority, urgency_from_prediction


class ClassificationEngine:
    def __init__(
        self,
        n_bands: int,
        weights: PriorityWeights | None = None,
        classify_every: int = 12,
        min_evidence: int = 6,
        evidence_target: int = 40,
    ) -> None:
        self.n_bands = int(n_bands)
        self.extractor = FeatureExtractor(n_bands, min_evidence=min_evidence)
        self.weights = weights or PriorityWeights()
        self.classify_every = int(classify_every)
        self.min_evidence = int(min_evidence)
        self.evidence_target = int(evidence_target)

        self._class: list[Classification] = [
            Classification(b, LABEL_UNCLASSIFIED, 0.15, "no data yet")
            for b in range(n_bands)
        ]
        self._beliefs = np.full(n_bands, 0.0)
        self._urgency = np.zeros(n_bands)
        self._priority = np.zeros(n_bands)
        self._breakdown: list[dict] = [{} for _ in range(n_bands)]
        self._periodicity_by_band: dict[int, dict] = {}
        self.t = 0

    def set_band_hints(self, band_info: list[dict]) -> None:
        self.extractor.set_band_hints(band_info)

    def set_weights(self, weights: PriorityWeights) -> None:
        self.weights = weights

    def reset(self) -> None:
        self.extractor.reset()
        self._class = [
            Classification(b, LABEL_UNCLASSIFIED, 0.15, "no data yet")
            for b in range(self.n_bands)
        ]
        self._beliefs[:] = 0.0
        self._urgency[:] = 0.0
        self._priority[:] = 0.0
        self._breakdown = [{} for _ in range(self.n_bands)]
        self._periodicity_by_band = {}
        self.t = 0

    # ------------------------------------------------------------ per tick
    def update(
        self,
        t: int,
        scanned_bands: np.ndarray,
        observations: np.ndarray,
        beliefs: np.ndarray,
        periodicity: list[dict],
        predicted_next_active: dict[int, float],
    ) -> None:
        self.t = t
        self._beliefs = np.asarray(beliefs, dtype=np.float64)
        self._periodicity_by_band = {e["band"]: e for e in (periodicity or [])}

        self.extractor.update(t, scanned_bands, observations, beliefs, periodicity)

        # Re-classify on a throttle with light hysteresis (anti-flicker).
        if t % self.classify_every == 0:
            for b in range(self.n_bands):
                feats = self.extractor.extract(b, self._periodicity_by_band)
                new = classify(feats, self.min_evidence, self.evidence_target)
                cur = self._class[b]
                # Keep the current label if the new one is a different, weaker
                # call; otherwise adopt the new classification.
                if new.label != cur.label and new.confidence < cur.confidence - 0.1:
                    continue
                self._class[b] = new

        # Priority every tick (belief moves fast).
        for b in range(self.n_bands):
            nxt = predicted_next_active.get(b)
            urg = urgency_from_prediction(nxt, t)
            self._urgency[b] = urg
            score, breakdown = compute_priority(
                float(self._beliefs[b]), self._class[b].confidence, urg, self.weights
            )
            self._priority[b] = score
            self._breakdown[b] = breakdown

    # ------------------------------------------------------------ readouts
    def snapshot(self) -> list[dict]:
        """Compact per-band classification + priority for the tick payload."""
        out = []
        for b in range(self.n_bands):
            c = self._class[b].to_dict()
            c["priority"] = round(float(self._priority[b]), 4)
            c["urgency"] = round(float(self._urgency[b]), 4)
            out.append(c)
        return out

    def band_detail(self, band: int) -> dict:
        """Full detail for the band popover + AI narration."""
        feats = self.extractor.extract(band, self._periodicity_by_band)
        c = self._class[band]
        return {
            "band": band,
            "belief": round(float(self._beliefs[band]), 4),
            "classification": c.to_dict(),
            "priority": round(float(self._priority[band]), 4),
            "priority_breakdown": self._breakdown[band],
            "features": feats.to_dict(),
            "periodicity": self._periodicity_by_band.get(band),
        }

    def top_bands(self, n: int = 6) -> list[dict]:
        order = np.argsort(-self._priority)[:n]
        return [self.band_detail(int(b)) for b in order]
