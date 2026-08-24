import unittest
import numpy as np
from src.common.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    asymmetric_weighted_loss,
)


class TestMetrics(unittest.TestCase):
    def test_accuracy(self):
        self.assertAlmostEqual(accuracy_score([1, 0, 1], [1, 0, 0]), 2 / 3)

    def test_f1(self):
        self.assertAlmostEqual(f1_score([1, 1, 0], [1, 0, 0]), 2 / 3)

    def test_rmse(self):
        self.assertAlmostEqual(root_mean_squared_error([1.0, 2.0], [2.0, 2.0]), np.sqrt(0.5))

    def test_r2(self):
        self.assertGreater(r2_score([1.0, 2.0, 3.0], [1.1, 1.9, 3.0]), 0.95)

    def test_asymmetric_loss(self):
        loss = asymmetric_weighted_loss([10.0], [8.0], under_prediction_weight=4.0)
        self.assertEqual(loss, 16.0)


if __name__ == "__main__":
    unittest.main()