
import tempfile
import unittest
from pathlib import Path

import numpy as np

from anyjev.medgemma_decider import MedGemmaDecider
from anyjev.calibrate.contextual import batch_prior, apply_contextual


class FakeBackend:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=np.float64)

    def score_image(self, index, prompt):
        return self.probabilities[int(index)]


class TestMedGemmaDecider(unittest.TestCase):

    def setUp(self):
        self.probs = np.array([
            [0.05, 0.10, 0.20, 0.25, 0.30, 0.10],
            [0.10, 0.15, 0.25, 0.20, 0.20, 0.10],
            [0.05, 0.05, 0.10, 0.20, 0.50, 0.10],
        ], dtype=np.float64)

        self.adapter = MedGemmaDecider(
            FakeBackend(self.probs),
            prompt="Score the clock",
            prior_strength=0.75,
        )

    def test_raw_probabilities(self):
        actual = self.adapter.predict_raw(range(3))
        np.testing.assert_allclose(actual, self.probs)

    def test_calibration(self):
        actual = self.adapter.calibrate(range(3))
        expected = batch_prior(self.probs)
        np.testing.assert_allclose(actual, expected)

    def test_prediction_matches_anyjev(self):
        prior = self.adapter.calibrate(range(3))
        result = self.adapter.predict(range(3))

        expected = apply_contextual(
            self.probs, prior ** 0.75
        )

        np.testing.assert_allclose(
            result["l0_probs"], expected
        )
        np.testing.assert_array_equal(
            result["l0_predictions"],
            expected.argmax(axis=1)
        )

    def test_independent_l0_calculation(self):
        self.adapter.prior = np.array(
            [0.10, 0.10, 0.10, 0.10, 0.50, 0.10]
        )
        result = self.adapter.predict([0])

        raw = self.probs[0]
        weights = raw / self.adapter.prior ** 0.75
        expected = weights / weights.sum()

        np.testing.assert_allclose(
            result["l0_probs"][0], expected, rtol=1e-12
        )
        self.assertEqual(
            int(result["l0_predictions"][0]),
            int(expected.argmax())
        )

    def test_rejects_invalid_probabilities(self):
        invalid = [
            np.array([[0.2, 0.3, 0.5]]),
            np.array([[0.1, 0.2, -0.1, 0.3, 0.4, 0.1]]),
            np.array([[np.nan, 0.2, 0.2, 0.2, 0.2, 0.2]]),
            np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]),
        ]
        for probabilities in invalid:
            with self.subTest(probabilities=probabilities):
                adapter = MedGemmaDecider(
                    FakeBackend(probabilities),
                    prompt="Score the clock",
                )
                with self.assertRaises(ValueError):
                    adapter.calibrate([0])

    def test_rejects_invalid_prior_strength(self):
        for strength in [-1, float("nan"), float("inf")]:
            with self.subTest(strength=strength):
                with self.assertRaises(ValueError):
                    MedGemmaDecider(
                        FakeBackend(self.probs),
                        prompt="Score the clock",
                        prior_strength=strength,
                    )

    def test_rejects_empty_calibration(self):
        with self.assertRaisesRegex(ValueError, "empty|nonempty"):
            self.adapter.calibrate([])

    def test_requires_calibration(self):
        with self.assertRaises(RuntimeError):
            self.adapter.predict(range(3))

    def test_rejects_unnormalized_saved_prior(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad_prior.npy"
            np.save(path, np.array([0.2] * 6))

            with self.assertRaises(ValueError):
                self.adapter.load_prior(path)

    def test_save_and_load_prior(self):
        original = self.adapter.calibrate(range(3)).copy()

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "prior.npy"
            self.adapter.save_prior(path)
            self.adapter.prior = None
            restored = self.adapter.load_prior(path)

        np.testing.assert_array_equal(original, restored)


if __name__ == "__main__":
    unittest.main()
