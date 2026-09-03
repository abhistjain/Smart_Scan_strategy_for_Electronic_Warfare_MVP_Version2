"""Per-band parametric feature extraction (v3 add-on, Section 2.2).

SCOPE (read Section 0 of the add-on spec): this is a *sensing / classification /
presentation* layer only. Nothing here selects, recommends, or models any
real-world weapon, interceptor, trajectory, or engagement. Features describe
observed SIGNAL BEHAVIOUR (regularity, agility, duty cycle) and feed only:
illustrative behaviour-pattern labels and a dashboard attention-sort score.

Everything is computed from operator-available information (what the receiver
observed when it scanned a band, plus the belief vector and the existing
periodicity output) - never from hidden ground truth - so the features are
realistic for an ES receiver.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class BandFeatures:
    band: int
    evidence: int                 # number of scans of this band in the window
    duty_cycle: float             # fraction of scans that observed activity
    onset_rate: float             # OFF->ON transitions per scan (burstiness)
    periodicity_strength: float   # 0..1 from the Lomb-Scargle / von Mises fit
    period: float                 # detected period in ticks (0 if none)
    period_trend: float           # slope of recent detected periods (<0 = tightening)
    bandwidth: float              # neighbour co-activity (proxy for wide occupancy)
    hop_rate: float               # 0..1 apparent movement to other bands
    amplitude_stability: float    # 0..1 (1 = very stable) - proxy in synthetic mode
    pulse_width_variance: float   # 0..1 normalised (0 = constant PW)

    def to_dict(self) -> dict:
        return {
            "band": self.band,
            "evidence": self.evidence,
            "duty_cycle": round(self.duty_cycle, 4),
            "onset_rate": round(self.onset_rate, 4),
            "periodicity_strength": round(self.periodicity_strength, 4),
            "period": round(self.period, 2),
            "period_trend": round(self.period_trend, 4),
            "bandwidth": round(self.bandwidth, 4),
            "hop_rate": round(self.hop_rate, 4),
            "amplitude_stability": round(self.amplitude_stability, 4),
            "pulse_width_variance": round(self.pulse_width_variance, 4),
        }


@dataclass
class _BandState:
    scans: deque = field(default_factory=lambda: deque(maxlen=160))       # 0/1 scanned
    obs: deque = field(default_factory=lambda: deque(maxlen=160))         # observed active on scan
    coactive: deque = field(default_factory=lambda: deque(maxlen=160))    # neighbour co-activity per tick
    hop_events: deque = field(default_factory=lambda: deque(maxlen=160))  # apparent hop to other band
    period_hist: deque = field(default_factory=lambda: deque(maxlen=24))  # (tick, period)
    last_obs: int = -1


class FeatureExtractor:
    """Maintains rolling per-band buffers and derives feature vectors.

    Call update() once per tick with the smart receiver's scans/observations,
    the current belief vector, and the periodicity summary; then extract(b) for
    any band.
    """

    def __init__(self, n_bands: int, window: int = 160, min_evidence: int = 6) -> None:
        self.n_bands = int(n_bands)
        self.window = int(window)
        self.min_evidence = int(min_evidence)
        self.state = [_BandState() for _ in range(n_bands)]
        # Static per-band hints from the environment (TSRD duty/PRI, synthetic
        # dwell) used as amplitude/PW proxies. Set via set_band_hints().
        self.band_hints: dict[int, dict] = {}

    def set_band_hints(self, band_info: list[dict]) -> None:
        for bi in band_info:
            self.band_hints[bi["index"]] = bi.get("stats", {}) or {}

    def reset(self) -> None:
        self.state = [_BandState() for _ in range(self.n_bands)]

    def update(
        self,
        t: int,
        scanned_bands: np.ndarray,
        observations: np.ndarray,
        beliefs: np.ndarray,
        periodicity: Optional[list[dict]] = None,
    ) -> None:
        scanned = np.asarray(scanned_bands, dtype=np.int64)
        obs = np.asarray(observations, dtype=np.int64)
        beliefs = np.asarray(beliefs, dtype=np.float64)
        scanned_set = {int(b): int(o) for b, o in zip(scanned, obs)}

        # Neighbour co-activity from the belief vector (operator-available).
        active_belief = beliefs > 0.5

        for b in range(self.n_bands):
            st = self.state[b]
            if b in scanned_set:
                o = scanned_set[b]
                st.scans.append(1)
                st.obs.append(o)
                # Onset / hop bookkeeping.
                if st.last_obs == 0 and o == 1:
                    # This band lit up; if a neighbour was the one previously
                    # active, treat it as an apparent hop INTO this band.
                    st.hop_events.append(1)
                else:
                    st.hop_events.append(0)
                st.last_obs = o
            else:
                st.scans.append(0)
                st.obs.append(0)
                st.hop_events.append(0)

            # Neighbour co-activity: fraction of adjacent bands also believed on.
            lo, hi = max(0, b - 1), min(self.n_bands - 1, b + 1)
            neigh = [j for j in range(lo, hi + 1) if j != b]
            if neigh and active_belief[b]:
                st.coactive.append(float(np.mean([active_belief[j] for j in neigh])))
            else:
                st.coactive.append(0.0)

        # Record detected periods (for the tightening-trend feature).
        if periodicity:
            for est in periodicity:
                b = est.get("band")
                if b is None or b >= self.n_bands:
                    continue
                if est.get("confidence", 0) >= 0.4 and est.get("period", 0) > 0:
                    self.state[b].period_hist.append((t, float(est["period"])))

    def extract(self, band: int, periodicity_by_band: dict[int, dict]) -> BandFeatures:
        st = self.state[band]
        scans = np.asarray(st.scans, dtype=np.float64)
        obs = np.asarray(st.obs, dtype=np.float64)
        evidence = int(scans.sum())

        active_count = float(obs.sum())
        onsets = float(np.asarray(st.hop_events, dtype=np.float64).sum())
        if evidence > 0:
            duty = float(active_count / evidence)
            onset_rate = float(onsets / evidence)
        else:
            duty = 0.0
            onset_rate = 0.0
        # Agility: fraction of active observations that are FRESH onsets. Isolated
        # single-tick bursts (hoppers) -> ~1.0; long steady runs (comms) -> ~0.
        burstiness = float(onsets / active_count) if active_count > 0 else 0.0

        est = periodicity_by_band.get(band)
        periodicity_strength = float(est["confidence"]) if est else 0.0
        period = float(est["period"]) if est else 0.0

        # Period trend: slope of recent (tick, period) points; <0 means the
        # inter-illumination interval is shrinking ("tightening pattern").
        period_trend = 0.0
        if len(st.period_hist) >= 4:
            ts = np.array([p[0] for p in st.period_hist], dtype=np.float64)
            ps = np.array([p[1] for p in st.period_hist], dtype=np.float64)
            if ts.max() > ts.min():
                period_trend = float(np.polyfit(ts, ps, 1)[0])

        bandwidth = float(np.mean(st.coactive)) if len(st.coactive) else 0.0
        hop_rate = float(np.clip(burstiness, 0.0, 1.0))

        # Amplitude/PW proxies from static hints (TSRD duty/PRI or synthetic).
        hints = self.band_hints.get(band, {})
        amplitude_stability = float(hints.get("amplitude_stability", 0.6))
        pw_var = hints.get("pulse_width_variance")
        if pw_var is None:
            # Proxy: periodic/steady -> low PW variance; hopping/bursty -> higher.
            pw_var = float(np.clip(hop_rate, 0.0, 1.0))
        pulse_width_variance = float(pw_var)

        return BandFeatures(
            band=band,
            evidence=evidence,
            duty_cycle=duty,
            onset_rate=onset_rate,
            periodicity_strength=periodicity_strength,
            period=period,
            period_trend=period_trend,
            bandwidth=bandwidth,
            hop_rate=hop_rate,
            amplitude_stability=amplitude_stability,
            pulse_width_variance=pulse_width_variance,
        )
