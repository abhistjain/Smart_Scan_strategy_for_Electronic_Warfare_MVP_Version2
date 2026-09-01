"""TSRD cache pipeline + loader tests (spec sections 2.2, 6.2, 11).

These use a small in-process fixture, NOT the live gated download, so CI needs no
network or HF credentials.
"""

import json
import os

import numpy as np

from data.prepare_tsrd_cache import (
    build_occupancy_grid,
    make_stub_pdws,
    _cluster_emitter_stats,
)
from sim.tsrd_environment import TSRDEnvironment, list_cached_samples, CACHE_DIR


def test_occupancy_grid_overlap_logic():
    # Two pulses: one in band 0 spanning slots 0-1, one in band 2 at slot 3.
    pdw = {
        "toa_us": np.array([0.0, 30.0]),
        "freq_mhz": np.array([50.0, 250.0]),   # mid band0 vs mid band2 (edges 0..400)
        "pulse_width_us": np.array([15.0, 2.0]),  # slot=10us -> first spans slot0,1
    }
    grid, meta = build_occupancy_grid(
        pdw, n_bands=4, slot_us=10.0, subband_mhz=(0.0, 400.0), duration_us=50.0
    )
    assert grid.shape == (5, 4)
    # Pulse 1: band 0, ToA 0, pw 15 -> slots 0 and 1 active.
    assert grid[0, 0] == 1 and grid[1, 0] == 1
    # Pulse 2: band 2, ToA 30 -> slot 3 active.
    assert grid[3, 2] == 1
    # Nothing else active.
    assert grid.sum() == 3


def test_stub_pipeline_produces_valid_grid():
    pdw = make_stub_pdws(seed=0)
    grid, meta = build_occupancy_grid(pdw, n_bands=24, slot_us=2000.0)
    assert grid.ndim == 2 and grid.shape[1] == 24
    assert grid.dtype == np.int8
    assert grid.sum() > 0
    stats = _cluster_emitter_stats(pdw, meta)
    assert isinstance(stats, list)


def test_list_samples_returns_list():
    samples = list_cached_samples()
    assert isinstance(samples, list)  # empty is fine (graceful Real-Data disable)


def test_environment_loads_and_steps_fixture():
    # Write a tiny fixture directly into the cache dir, load, step, clean up.
    cache = os.path.abspath(CACHE_DIR)
    os.makedirs(cache, exist_ok=True)
    sid = 987
    grid = np.zeros((10, 5), dtype=np.int8)
    grid[::2, 1] = 1  # band 1 active on even slots -> periodic
    grid[3, 4] = 1
    npz = os.path.join(cache, f"tsrd_sample_{sid}.npz")
    js = os.path.join(cache, f"tsrd_sample_{sid}.json")
    np.savez_compressed(npz, grid=grid)
    with open(js, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "n_bands": 5, "n_slots": 10, "slot_us": 1000.0,
                "sample_id": sid, "stub": True, "emitter_stats": [],
                "band_edges_mhz": [0, 100, 200, 300, 400, 500],
            },
            fh,
        )
    try:
        env = TSRDEnvironment(sample_id=sid, p_miss=0.1, p_fa=0.05, loop=True)
        assert env.n_bands == 5
        s0 = env.ground_truth_state
        assert s0[1] == 1
        env.step()  # slot 1 -> band 1 off
        assert env.ground_truth_state[1] == 0
        # Looping: after n_slots steps we should be back near the start.
        for _ in range(9):
            env.step()
        assert env.t == 10
        assert len(env.band_info()) == 5
    finally:
        os.remove(npz)
        os.remove(js)
