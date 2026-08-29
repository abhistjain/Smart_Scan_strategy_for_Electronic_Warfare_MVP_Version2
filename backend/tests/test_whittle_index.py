"""Whittle index property tests (spec section 6.5)."""

import numpy as np

from algo.whittle_index import WhittleIndexEngine, whittle_index_scalar


def test_monotonic_in_belief_positively_correlated():
    # p11 > p01 is the positively correlated regime where the index MUST be
    # monotonically increasing in belief - the property that makes the policy
    # near-optimal.
    for p01, p10 in [(0.05, 0.3), (0.02, 0.5), (0.2, 0.25), (0.1, 0.4)]:
        p11 = 1 - p10
        assert p11 >= p01
        ws = [whittle_index_scalar(w, p01, p10, beta=0.99) for w in np.linspace(0, 1, 501)]
        diffs = np.diff(ws)
        assert np.all(diffs >= -1e-9), f"non-monotonic for p01={p01}, p10={p10}"


def test_monotonic_in_belief_negatively_correlated():
    # Thompson samples can land in the negatively correlated regime early on;
    # the closed form must still be well-behaved (monotone) there.
    for p01, p10 in [(0.6, 0.6), (0.7, 0.5), (0.8, 0.4)]:
        p11 = 1 - p10
        assert p11 < p01
        ws = [whittle_index_scalar(w, p01, p10, beta=0.99) for w in np.linspace(0, 1, 501)]
        diffs = np.diff(ws)
        assert np.all(diffs >= -1e-9), f"non-monotonic for p01={p01}, p10={p10}"


def test_boundary_values():
    # W(0)=0 and W(1)=1 (index equals belief at the extremes).
    assert np.isclose(whittle_index_scalar(0.0, 0.05, 0.3), 0.0, atol=1e-6)
    assert np.isclose(whittle_index_scalar(1.0, 0.05, 0.3), 1.0, atol=1e-6)


def test_continuity_across_regions():
    # No jumps at the region boundaries p01, w_o, p11 for a positively
    # correlated channel.
    p01, p10 = 0.1, 0.3
    ws = np.linspace(0, 1, 2000)
    vals = np.array([whittle_index_scalar(w, p01, p10, beta=0.99) for w in ws])
    assert np.max(np.abs(np.diff(vals))) < 0.02


def test_engine_vectorised_matches_scalar():
    eng = WhittleIndexEngine(n_bands=5, beta=0.99)
    belief = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    p01 = np.full(5, 0.05)
    p10 = np.full(5, 0.3)
    arr = eng.index_array(belief, p01, p10)
    for i in range(5):
        assert np.isclose(arr[i], whittle_index_scalar(belief[i], 0.05, 0.3, beta=0.99))


def test_reward_scaling():
    a = whittle_index_scalar(0.6, 0.1, 0.3, reward=1.0)
    b = whittle_index_scalar(0.6, 0.1, 0.3, reward=2.5)
    assert np.isclose(b, 2.5 * a)
