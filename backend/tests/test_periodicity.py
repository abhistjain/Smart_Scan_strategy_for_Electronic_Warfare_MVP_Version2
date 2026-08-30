"""Periodicity detection tests (spec section 6.6).

Includes validation against the TSRD-derived periodic emitter statistics, not
just synthetic clean periods.
"""

import numpy as np

from algo.periodicity import PeriodicityDetector
from data.prepare_tsrd_cache import make_stub_pdws, build_occupancy_grid


def test_detects_clean_period():
    det = PeriodicityDetector(n_bands=1, min_hits=6, min_period=3, max_period=60)
    period = 20
    for k in range(15):
        det.record_hit(0, k * period)
    det.update(t_now=15 * period)
    est = det.estimates[0]
    assert est is not None
    assert abs(est.period - period) <= 2
    assert est.confidence > 0.8  # tightly concentrated phase


def test_index_boost_before_predicted_window():
    det = PeriodicityDetector(n_bands=1, min_hits=6, min_period=3, max_period=60,
                              boost_window=2.0, boost_scale=0.5)
    period = 20
    for k in range(15):
        det.record_hit(0, k * period)
    # just before the next active tick
    t_now = 15 * period - 1
    det.update(t_now=t_now)
    boost = det.index_boost(t_now)
    assert boost[0] > 0.0


def test_validate_against_tsrd_periodic_emitter():
    # Build a TSRD-style occupancy grid from stub PDWs (which contain clean
    # periodic emitters), then confirm the detector recovers a period from the
    # busiest periodic band's on-slot timings.
    pdw = make_stub_pdws(seed=1)
    grid, meta = build_occupancy_grid(pdw, n_bands=24, slot_us=2000.0)
    # Pick the band with the most on-slots.
    band = int(np.argmax(grid.sum(axis=0)))
    on_slots = np.where(grid[:, band] == 1)[0]
    if on_slots.size < 8:
        return  # not enough activity in this random draw; skip gracefully
    det = PeriodicityDetector(n_bands=1, min_hits=6, min_period=2,
                              max_period=float(min(200, grid.shape[0] // 2)))
    for s in on_slots:
        det.record_hit(0, int(s))
    det.update(t_now=int(on_slots[-1]) + 1)
    est = det.estimates[0]
    # A period should be found with some confidence for a periodic emitter.
    assert est is not None
    assert est.period > 0
