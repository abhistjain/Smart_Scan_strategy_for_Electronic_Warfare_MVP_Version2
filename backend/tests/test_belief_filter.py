"""Belief filter validated against a hand-computed 3-step toy example (6.3)."""

import numpy as np

from algo.belief_filter import BeliefFilter


def test_three_step_toy_example():
    # Observation model: p_miss=0.2, p_fa=0.1  =>
    #   P(o=1|ON)=0.8, P(o=0|ON)=0.2, P(o=1|OFF)=0.1, P(o=0|OFF)=0.9
    # Transitions: p01=0.1, p10=0.2 (p11=0.8). Single band, b0=0.5.
    bf = BeliefFilter(n_bands=1, p_miss=0.2, p_fa=0.1, init_belief=0.5)
    p01 = np.array([0.1])
    p10 = np.array([0.2])

    # --- Step 1: predict then scan, observe o=1 -----------------------------
    # predict: 0.5*0.8 + 0.5*0.1 = 0.45
    # update o=1: (0.8*0.45)/(0.8*0.45 + 0.1*0.55) = 0.36/0.415 = 0.86746988
    b1 = bf.step(p01, p10, scanned_bands=np.array([0]), observations=np.array([1]))
    assert np.isclose(b1[0], 0.36 / 0.415, atol=1e-9)

    # --- Step 2: predict then scan, observe o=0 -----------------------------
    # predict: 0.86746988*0.8 + 0.13253012*0.1 = 0.70722892
    # update o=0: (0.2*0.70722892)/(0.2*0.70722892 + 0.9*0.29277108) = 0.34930...
    b2 = bf.step(p01, p10, scanned_bands=np.array([0]), observations=np.array([0]))
    expected_b2 = 0.34930030959752316
    assert np.isclose(b2[0], expected_b2, atol=1e-6)

    # --- Step 3: NOT scanned -> prediction only -----------------------------
    # predict: 0.34930031*0.8 + 0.65069969*0.1 = 0.34451049
    b3 = bf.step(p01, p10, scanned_bands=np.array([], dtype=int),
                 observations=np.array([], dtype=int))
    expected_b3 = expected_b2 * 0.8 + (1 - expected_b2) * 0.1
    assert np.isclose(b3[0], expected_b3, atol=1e-6)


def test_certain_observation_pushes_belief_toward_truth():
    bf = BeliefFilter(n_bands=1, p_miss=0.0, p_fa=0.0, init_belief=0.5)
    b = bf.step(np.array([0.2]), np.array([0.2]),
                scanned_bands=np.array([0]), observations=np.array([1]))
    # With no sensor noise, a positive observation implies ON with certainty.
    assert b[0] > 0.999


def test_unscanned_band_relaxes_to_stationary():
    # With repeated prediction only, belief converges to stationary p01/(p01+p10).
    bf = BeliefFilter(n_bands=1, p_miss=0.1, p_fa=0.05, init_belief=0.9)
    p01, p10 = np.array([0.1]), np.array([0.3])
    for _ in range(500):
        bf.predict(p01, p10)
    assert np.isclose(bf.belief[0], 0.1 / (0.1 + 0.3), atol=1e-3)
