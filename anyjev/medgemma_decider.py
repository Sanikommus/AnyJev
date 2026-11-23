"""Multimodal MedGemma decision adapter using AnyJev L0 calibration."""

import numpy as np

from anyjev.calibrate.contextual import batch_prior, apply_contextual


class MedGemmaDecider:
    def __init__(self, backend, prompt, prior_strength=0.75):
        self.backend = backend
        self.prompt = prompt
        self.prior_strength = float(prior_strength)
        if not np.isfinite(self.prior_strength) or self.prior_strength < 0:
            raise ValueError(
                "prior_strength must be finite and nonnegative."
            )
        self.prior = None

    def predict_raw(self, image_paths):
        """Return [N, 6] raw score probabilities."""
        probabilities = np.asarray([
            self.backend.score_image(path, self.prompt)
            for path in image_paths
        ], dtype=np.float64)

        if probabilities.size == 0:
            raise ValueError("Image list must be nonempty.")

        if probabilities.ndim != 2 or probabilities.shape[1] != 6:
            raise ValueError("Expected probability array of shape [N, 6].")

        if not np.all(np.isfinite(probabilities)):
            raise ValueError("Probabilities must be finite.")

        if np.any(probabilities < 0):
            raise ValueError("Probabilities must be nonnegative.")

        if not np.allclose(
            probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6
        ):
            raise ValueError("Each probability distribution must sum to 1.")

        return probabilities

    def calibrate(self, image_paths):
        """Estimate a fixed batch prior on separate calibration images."""
        probabilities = self.predict_raw(image_paths)
        if probabilities.size == 0:
            raise ValueError("Calibration image list must be nonempty.")
        self.prior = batch_prior(probabilities)
        return self.prior

    def save_prior(self, path):
        """Save the full-precision calibration prior."""
        if self.prior is None:
            raise RuntimeError("No calibration prior available.")
        np.save(path, self.prior)

    def load_prior(self, path):
        """Load a previously saved calibration prior."""
        prior = np.load(path, allow_pickle=False)
        if prior.shape != (6,):
            raise ValueError("Expected a prior with six score probabilities.")
        if not np.all(np.isfinite(prior)) or np.any(prior <= 0):
            raise ValueError("Calibration prior must be finite and positive.")
        if not np.isclose(prior.sum(), 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError("Calibration prior probabilities must sum to 1.")
        self.prior = prior
        return self.prior

    def predict(self, image_paths):
        """Return raw and L0 predictions with their probabilities."""
        if self.prior is None:
            raise RuntimeError("Call calibrate() before predict().")

        raw = self.predict_raw(image_paths)
        l0 = apply_contextual(raw, np.power(self.prior, self.prior_strength))

        return {
            "raw_probs": raw,
            "l0_probs": l0,
            "raw_predictions": raw.argmax(axis=1),
            "l0_predictions": l0.argmax(axis=1),
        }
