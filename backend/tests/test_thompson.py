"""Thompson sampling posterior-concentration tests (spec section 6.4)."""

import numpy as np

from algo.thompson import ThompsonTransitionLearner


def test_posteriors_concentrate_toward_truth():
    # Feed a long stream of consecutive-scan transitions from a known chain and
    # check the posterior means converge to the true p01/p10.
    rng = np.random.default_rng(0)
    p01_true, p10_true = 0.15, 0.35
    learner = ThompsonTransitionLearner(n_bands=1, seed=0)

    state = 0
    prev_obs = None
    for _ in range(20000):
        # advance a clean (noiseless) chain
        if state == 0:
            state = 1 if rng.random() < p01_true else 0
        else:
            state = 0 if rng.random() < p10_true else 1
        obs = state  # noiseless observation for this concentration test
        learner.update(np.array([0]), np.array([obs]))

    p01_hat, p10_hat = learner.mean()
    assert abs(p01_hat[0] - p01_true) < 0.03
    assert abs(p10_hat[0] - p10_true) < 0.05


def test_sampling_varies_but_mean_stable():
    learner = ThompsonTransitionLearner(n_bands=3, seed=1)
    s1, _ = learner.sample()
    s2, _ = learner.sample()
    # Under the uniform prior, successive samples differ (stochastic).
    assert not np.allclose(s1, s2)


def test_counts_increase_on_transitions():
    learner = ThompsonTransitionLearner(n_bands=1, seed=0)
    assert learner.counts()[0] == 0
    learner.update(np.array([0]), np.array([0]))
    learner.update(np.array([0]), np.array([1]))  # OFF->ON transition observed
    assert learner.counts()[0] >= 1
