import unittest

from anyjev.backends.medgemma import MedGemmaBackend


class FakeTokenizer:
    def __init__(self, mapping):
        self.mapping = mapping

    def encode(self, text, add_special_tokens=False):
        return self.mapping[text]


class TestMedGemmaBackend(unittest.TestCase):

    def make_backend(self, mapping):
        # Avoid loading the actual MedGemma model.
        backend = MedGemmaBackend.__new__(MedGemmaBackend)
        backend.tokenizer = FakeTokenizer(mapping)
        return backend

    def test_valid_score_tokens(self):
        mapping = {str(i): [100 + i] for i in range(6)}
        backend = self.make_backend(mapping)

        result = backend.resolve_score_tokens(
            [str(i) for i in range(6)]
        )

        self.assertEqual(result, [100, 101, 102, 103, 104, 105])

    def test_rejects_multitoken_labels(self):
        mapping = {"0": [100], "1": [101, 102]}
        backend = self.make_backend(mapping)

        with self.assertRaises(ValueError):
            backend.resolve_score_tokens(["0", "1"])

    def test_rejects_duplicate_token_ids(self):
        mapping = {"0": [100], "1": [100]}
        backend = self.make_backend(mapping)

        with self.assertRaises(ValueError):
            backend.resolve_score_tokens(["0", "1"])

    def test_rejects_empty_labels(self):
        backend = self.make_backend({})

        with self.assertRaises(ValueError):
            backend.resolve_score_tokens([])


if __name__ == "__main__":
    unittest.main()
