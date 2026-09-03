"""Transparent rule-based emitter *behaviour-pattern* classifier (Section 2.3).

FRAMING (this exact text is surfaced in the UI, Section 2.1):

    "Category labels below describe SIGNAL BEHAVIOUR PATTERNS observed in this
    simulation (e.g. pulse regularity, frequency agility, duty cycle) and are
    illustrative analogues loosely inspired by general EW literature. They are
    NOT a validated identification-friend-or-foe (IFF) system and must not be
    treated as real platform identification."

The classifier is a small, explainable decision table over the features from
features.py. Every classification logs a MATCHED_RULE string so the exact reason
can be shown in a debug tooltip. There is deliberately NO weapon/engagement
logic anywhere (add-on Section 0).
"""

from __future__ import annotations

from dataclasses import dataclass

from .features import BandFeatures

DISCLAIMER = (
    "Category labels describe signal behaviour patterns observed in this "
    "simulation (pulse regularity, frequency agility, duty cycle) and are "
    "illustrative analogues loosely inspired by general EW literature. They are "
    "NOT a validated IFF system and must not be treated as real platform "
    "identification."
)

# Canonical labels.
LABEL_COMMS = "Steady Comms-Like Emission"
LABEL_SCANNING = "Scanning/Rotating-Pattern Emission"
LABEL_HOPPING = "Frequency-Agile / Hopping Emission"
LABEL_TIGHTENING = "Tightening-Pattern Emission"
LABEL_UNCLASSIFIED = "Unclassified"

# Short chip labels + colours for the UI.
LABEL_META = {
    LABEL_COMMS: {"short": "Comms-Like", "color": "#34D399"},
    LABEL_SCANNING: {"short": "Scanning-Pattern", "color": "#22D3EE"},
    LABEL_HOPPING: {"short": "Agile/Hopping", "color": "#F5A623"},
    LABEL_TIGHTENING: {"short": "Tightening-Pattern", "color": "#EF4444"},
    LABEL_UNCLASSIFIED: {"short": "Unclassified", "color": "#64748B"},
}

# Competition-only *illustrative analogues* for the dashboard AI block.
# These are a presentation mapping of the behaviour labels above — NOT IFF,
# NOT real platform ID, and they never name a real airframe or munition.
ANALOGUE_META = {
    LABEL_SCANNING: {
        "key": "fighter",
        "short": "Fighter-like",
        "title": "Fighter-like search analogue",
        "why": "Regular scan cadence and moderate duty — resembles a rotating airborne search pattern in general EW literature.",
        "glyph": "✈",
        "color": "#22D3EE",
    },
    LABEL_HOPPING: {
        "key": "uav",
        "short": "UAV-like",
        "title": "UAV / drone-like agile analogue",
        "why": "Low duty and high hop rate — resembles a frequency-agile small-platform emitter.",
        "glyph": "⬡",
        "color": "#F5A623",
    },
    LABEL_TIGHTENING: {
        "key": "missile",
        "short": "Missile-like",
        "title": "Missile-like tightening analogue",
        "why": "Detected period is shrinking — resembles a scan becoming more frequent (behaviour only, not a lock-on).",
        "glyph": "◆",
        "color": "#EF4444",
    },
    LABEL_COMMS: {
        "key": "comms",
        "short": "Datalink-like",
        "title": "Datalink / comms analogue",
        "why": "High duty and stable frequency — resembles a continuous comms or datalink emission.",
        "glyph": "▣",
        "color": "#34D399",
    },
    LABEL_UNCLASSIFIED: {
        "key": "unknown",
        "short": "Unknown",
        "title": "Unclassified",
        "why": "Not enough clean evidence for an analogue.",
        "glyph": "·",
        "color": "#64748B",
    },
}


@dataclass
class Classification:
    band: int
    label: str
    confidence: float
    matched_rule: str

    def to_dict(self) -> dict:
        meta = LABEL_META.get(self.label, LABEL_META[LABEL_UNCLASSIFIED])
        analogue = ANALOGUE_META.get(self.label, ANALOGUE_META[LABEL_UNCLASSIFIED])
        return {
            "band": self.band,
            "label": self.label,
            "short": meta["short"],
            "color": meta["color"],
            "confidence": round(self.confidence, 3),
            "matched_rule": self.matched_rule,
            "analogue_key": analogue["key"],
            "analogue_short": analogue["short"],
            "analogue_title": analogue["title"],
            "analogue_why": analogue["why"],
            "analogue_glyph": analogue["glyph"],
            "analogue_color": analogue["color"],
        }


def _evidence_factor(evidence: int, target: int = 40) -> float:
    """Confidence grows with observed evidence, saturating at `target` scans."""
    return float(min(1.0, evidence / max(target, 1)))


def classify(
    f: BandFeatures,
    min_evidence: int = 6,
    evidence_target: int = 40,
) -> Classification:
    """Map a feature vector to a behaviour-pattern label + confidence.

    The order of checks matters: the higher-priority "tightening" pattern is
    tested before the generic scanning pattern.
    """
    ev = _evidence_factor(f.evidence, evidence_target)

    if f.evidence < min_evidence:
        return Classification(
            f.band, LABEL_UNCLASSIFIED, confidence=0.15,
            matched_rule=f"evidence<{min_evidence} (have {f.evidence})",
        )

    # A period of ~1 tick just means "active almost every tick" (an always-on
    # emitter), not a genuine scan cadence - ignore it so steady emitters aren't
    # blocked from the comms-like rule by trivial periodicity.
    meaningful_period = f.period > 1.5
    strong_period = meaningful_period and f.periodicity_strength >= 0.55
    some_period = meaningful_period and f.periodicity_strength >= 0.4

    # (1) Tightening pattern: periodic AND the inter-illumination interval is
    #     measurably shrinking. Phrased behaviour-first; flagged higher priority.
    if some_period and f.period_trend < -0.05 and f.duty_cycle < 0.6:
        rule_strength = min(1.0, abs(f.period_trend) / 0.2) * 0.5 + 0.5 * f.periodicity_strength
        return Classification(
            f.band, LABEL_TIGHTENING,
            confidence=round(ev * (0.6 + 0.4 * rule_strength), 3),
            matched_rule="some_period AND period_trend<-0.05 AND duty<0.6",
        )

    # (2) Scanning / rotating pattern: strong periodicity, moderate duty, and
    #     narrow occupancy (low neighbour co-activity => single band).
    if strong_period and 0.03 <= f.duty_cycle <= 0.6 and f.bandwidth < 0.5:
        return Classification(
            f.band, LABEL_SCANNING,
            confidence=round(ev * (0.5 + 0.5 * f.periodicity_strength), 3),
            matched_rule="strong_period AND 0.03<=duty<=0.6 AND bandwidth<0.5",
        )

    # (3) Frequency-agile / hopping: low duty, mostly isolated bursts (high
    #     agility), weak periodicity.
    if f.duty_cycle < 0.35 and f.hop_rate >= 0.6 and not strong_period:
        return Classification(
            f.band, LABEL_HOPPING,
            confidence=round(ev * (0.4 + 0.6 * f.hop_rate), 3),
            matched_rule="duty<0.35 AND hop_rate>=0.6 AND not strong_period",
        )

    # (4) Steady comms-like: high duty, low burstiness, weak periodicity.
    if f.duty_cycle >= 0.6 and f.onset_rate < 0.2 and not some_period:
        return Classification(
            f.band, LABEL_COMMS,
            confidence=round(ev * (0.5 + 0.5 * f.duty_cycle), 3),
            matched_rule="duty>=0.6 AND onset_rate<0.2 AND not some_period",
        )

    # (5) Fallback.
    return Classification(
        f.band, LABEL_UNCLASSIFIED,
        confidence=round(0.25 * ev, 3),
        matched_rule="no rule matched cleanly",
    )
