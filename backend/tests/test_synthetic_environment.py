"""Synthetic environment statistical tests (spec section 6.1)."""

import numpy as np

from sim.synthetic_environment import SyntheticEnvironment


def test_reproducible_with_seed():
    e1 = SyntheticEnvironment(n_bands=16, seed=42)
    e2 = SyntheticEnvironment(n_bands=16, seed=42)
    for _ in range(200):
        s1 = e1.step()
        s2 = e2.step()
        assert np.array_equal(s1, s2)


def test_markov_transition_frequencies():
    # For a purely-markov band, empirical transition frequencies should match
    # the configured p01/p10 within sampling error.
    env = SyntheticEnvironment(
        n_bands=1, seed=1, emitter_mix={"markov": 1.0, "periodic": 0, "hopper": 0, "quiet": 0}
    )
    spec = env.specs[0]
    assert spec.kind == "markov"
    p01, p10 = spec.p01, spec.p10

    states = [env.ground_truth_state[0]]
    for _ in range(60000):
        states.append(env.step()[0])
    states = np.array(states)

    off = states[:-1] == 0
    on = states[:-1] == 1
    emp_p01 = np.mean(states[1:][off] == 1)
    emp_p10 = np.mean(states[1:][on] == 0)
    assert abs(emp_p01 - p01) < 0.03
    assert abs(emp_p10 - p10) < 0.03


def test_periodic_emitter_period():
    env = SyntheticEnvironment(
        n_bands=1, seed=3,
        emitter_mix={"markov": 0, "periodic": 1.0, "hopper": 0, "quiet": 0},
    )
    spec = env.specs[0]
    assert spec.kind == "periodic"
    on_ticks = []
    for t in range(1, 2000):
        s = env.step()
        if s[0] == 1:
            on_ticks.append(t)
    # Gaps between consecutive active windows should cluster near the period.
    on_ticks = np.array(on_ticks)
    if on_ticks.size > 3:
        gaps = np.diff(on_ticks)
        big_gaps = gaps[gaps > 1]  # gaps between windows (ignore within-dwell)
        # Median inter-window gap ~ period (allow jitter tolerance).
        assert abs(np.median(big_gaps) - spec.period) <= 3


def test_hopper_single_active_band():
    env = SyntheticEnvironment(
        n_bands=6, seed=5,
        emitter_mix={"markov": 0, "periodic": 0, "hopper": 1.0, "quiet": 0},
    )
    # Every band is a hopper; within each hop group exactly one band is ON.
    for _ in range(100):
        env.step()
        for gid, bands in env.hop_groups.items():
            assert sum(int(env.ground_truth_state[b]) for b in bands) == 1


def test_observation_noise_model():
    env = SyntheticEnvironment(n_bands=1, seed=7, p_miss=0.2, p_fa=0.1,
                               emitter_mix={"markov": 1.0, "periodic": 0, "hopper": 0, "quiet": 0})
    rng = np.random.default_rng(0)
    on_obs, off_obs = [], []
    for _ in range(40000):
        env.step()
        truth = env.ground_truth_state[0]
        o = env.observe(np.array([0]), rng)[0]
        (on_obs if truth == 1 else off_obs).append(o)
    if on_obs:
        assert abs(np.mean(on_obs) - 0.8) < 0.03   # 1 - p_miss
    if off_obs:
        assert abs(np.mean(off_obs) - 0.1) < 0.03   # p_fa
