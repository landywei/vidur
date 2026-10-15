"""Random-forest training does not depend on the global random state."""

import unittest

import numpy as np

from vidur.execution_time_predictor.random_forrest_execution_time_predictor import (
    TRAINING_SEED,
    RandomForrestExecutionTimePredictor,
)


def trained(draws):
    """A forest fitted after `draws` draws from numpy's global generator."""
    np.random.seed(7)
    np.random.random(draws)
    x = np.arange(200, dtype=float).reshape(-1, 1)
    y = np.sin(x[:, 0] / 9) + x[:, 0] / 50
    model = RandomForrestExecutionTimePredictor.__new__(
        RandomForrestExecutionTimePredictor
    )
    forest = model._get_estimator().set_params(n_estimators=8, max_depth=4)
    return forest.fit(x, y).predict(x + 0.5)


class PredictorSeedTests(unittest.TestCase):
    def test_training_ignores_earlier_random_draws(self):
        self.assertTrue(np.array_equal(trained(0), trained(13)))

    def test_estimator_carries_the_training_seed(self):
        model = RandomForrestExecutionTimePredictor.__new__(
            RandomForrestExecutionTimePredictor
        )
        self.assertEqual(model._get_estimator().random_state, TRAINING_SEED)


if __name__ == "__main__":
    unittest.main()
