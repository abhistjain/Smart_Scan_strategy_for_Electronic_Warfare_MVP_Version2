"""Threat Priority Score (Section 2.4).

A single relative-urgency number per band, used ONLY to sort the operator's
dashboard belief panel (highest first). It does NOT feed any weapon,
interceptor, engagement, or response logic - there is none in this app
(add-on Section 0). The persistent UI disclaimer must be shown wherever this
score appears.

    priority_i = w_belief   * belief_i
               + w_conf     * classification_confidence_i
               + w_urgency  * urgency_i

where urgency_i is derived from the EXISTING periodicity module's predicted time
to the next active window (sooner -> more urgent). No new prediction logic is
introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PriorityWeights:
    w_belief: float = 0.5
    w_conf: float = 0.2
    w_urgency: float = 0.3

    def to_dict(self) -> dict:
        return {
            "w_belief": self.w_belief,
            "w_conf": self.w_conf,
            "w_urgency": self.w_urgency,
        }


def urgency_from_prediction(
    next_active_tick: float | None, t_now: int, horizon: float = 20.0
) -> float:
    """Map 'ticks until next predicted active window' to urgency in [0, 1].

    Sooner windows are more urgent. Beyond `horizon` ticks -> 0 urgency.
    """
    if next_active_tick is None:
        return 0.0
    dt = next_active_tick - t_now
    if dt < 0:
        dt = 0.0
    if dt >= horizon:
        return 0.0
    return float(1.0 - dt / horizon)


def compute_priority(
    belief: float,
    confidence: float,
    urgency: float,
    weights: PriorityWeights,
) -> tuple[float, dict]:
    """Return (score, breakdown) so the UI can show a transparent breakdown."""
    b = weights.w_belief * belief
    c = weights.w_conf * confidence
    u = weights.w_urgency * urgency
    total = b + c + u
    breakdown = {
        "belief_term": round(b, 4),
        "confidence_term": round(c, 4),
        "urgency_term": round(u, 4),
        "belief": round(belief, 4),
        "confidence": round(confidence, 4),
        "urgency": round(urgency, 4),
    }
    return round(total, 4), breakdown
