"""Feature-extraction + rule-based classifier tests (v3 add-on Sections 2, 5).

Uses hand-crafted bands with known behaviour (clearly periodic, clearly hopping,
clearly steady) and asserts both that the extracted features look sane and that
each maps to the expected behaviour-pattern label.
"""

import numpy as np

from classification.classifier import (
    LABEL_COMMS,
    LABEL_HOPPING,
    LABEL_SCANNING,
    LABEL_TIGHTENING,
    LABEL_UNCLASSIFIED,
    classify,
)
from classification.features import FeatureExtractor
from classification.priority_score import (
    PriorityWeights,
    compute_priority,
    urgency_from_prediction,
)


def _run(extractor: FeatureExtractor, band: int, pattern, period_conf=0.0, period=0.0,
         ticks=120):
    """Drive one band through `ticks` steps. `pattern(t)` -> observed 0/1 when
    scanned (we scan this band every tick for a clean test)."""
    for t in range(1, ticks + 1):
        obs = pattern(t)
        beliefs = np.zeros(extractor.n_bands)
        beliefs[band] = 1.0 if obs else 0.0
        periodicity = (
            [{"band": band, "period": period, "confidence": period_conf}]
            if period_conf > 0
            else []
        )
        extractor.update(t, np.array([band]), np.array([obs]), beliefs, periodicity)


def test_steady_comms_like():
    ex = FeatureExtractor(n_bands=1)
    _run(ex, 0, lambda t: 1)  # always active -> high duty, no bursts
    f = ex.extract(0, {})
    assert f.duty_cycle > 0.9
    assert f.onset_rate < 0.1
    c = classify(f)
    assert c.label == LABEL_COMMS
    assert c.confidence > 0.3


def test_scanning_periodic():
    ex = FeatureExtractor(n_bands=1)
    period = 10
    # Active for 2 of every 10 ticks -> moderate duty, strong periodicity.
    _run(ex, 0, lambda t: 1 if (t % period) < 2 else 0,
         period_conf=0.9, period=period)
    f = ex.extract(0, {0: {"band": 0, "period": period, "confidence": 0.9}})
    assert 0.05 <= f.duty_cycle <= 0.6
    assert f.periodicity_strength >= 0.55
    c = classify(f)
    assert c.label == LABEL_SCANNING


def test_frequency_hopping():
    ex = FeatureExtractor(n_bands=1)
    # Rare, isolated bursts -> low duty, high onset rate, no periodicity.
    _run(ex, 0, lambda t: 1 if (t % 7 == 0) else 0, period_conf=0.0)
    f = ex.extract(0, {})
    assert f.duty_cycle < 0.35
    assert f.hop_rate >= 0.6
    c = classify(f)
    assert c.label == LABEL_HOPPING


def test_tightening_pattern_flagged():
    ex = FeatureExtractor(n_bands=1)
    # Feed a shrinking period history (decreasing period over time).
    for i, t in enumerate(range(1, 60)):
        beliefs = np.zeros(1)
        period = max(4.0, 20.0 - i * 0.3)  # decreasing -> negative trend
        ex.update(t, np.array([0]), np.array([1 if t % 5 == 0 else 0]), beliefs,
                  [{"band": 0, "period": period, "confidence": 0.8}])
    f = ex.extract(0, {0: {"band": 0, "period": 6.0, "confidence": 0.8}})
    assert f.period_trend < 0
    c = classify(f)
    assert c.label == LABEL_TIGHTENING


def test_insufficient_evidence_is_unclassified():
    ex = FeatureExtractor(n_bands=1)
    ex.update(1, np.array([0]), np.array([1]), np.array([1.0]), [])
    f = ex.extract(0, {})
    c = classify(f)
    assert c.label == LABEL_UNCLASSIFIED


def test_priority_score_and_urgency():
    w = PriorityWeights(0.5, 0.2, 0.3)
    # Sooner predicted window -> higher urgency.
    u_soon = urgency_from_prediction(next_active_tick=102, t_now=100, horizon=20)
    u_late = urgency_from_prediction(next_active_tick=118, t_now=100, horizon=20)
    assert u_soon > u_late
    score, breakdown = compute_priority(belief=0.8, confidence=0.6, urgency=u_soon, weights=w)
    assert score > 0
    assert set(breakdown) >= {"belief_term", "confidence_term", "urgency_term"}


def test_matched_rule_is_explainable():
    ex = FeatureExtractor(n_bands=1)
    _run(ex, 0, lambda t: 1)
    c = classify(ex.extract(0, {}))
    assert isinstance(c.matched_rule, str) and len(c.matched_rule) > 0
