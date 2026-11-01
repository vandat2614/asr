"""Tests for src/asr/metrics.py. Run: python3 -m unittest discover -s tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asr.metrics import corpus_cer, corpus_wer, edit_distance  # noqa: E402
from asr.normalize import normalize_basic, normalize_text  # noqa: E402


class MetricsTest(unittest.TestCase):
    def test_edit_distance(self):
        self.assertEqual(edit_distance(["a", "b"], ["a", "b"]), 0)
        self.assertEqual(edit_distance(["a", "b"], ["a", "c"]), 1)
        self.assertEqual(edit_distance(["a", "b"], ["a"]), 1)
        self.assertEqual(edit_distance([], ["a", "b"]), 2)

    def test_wer_values(self):
        ident = lambda s: s  # noqa: E731
        self.assertEqual(corpus_wer(["a b c d"], ["a b c d"], ident), 0.0)
        self.assertEqual(corpus_wer(["a b c d"], ["a b x d"], ident), 0.25)
        self.assertEqual(corpus_wer(["a b c d"], ["a b c"], ident), 0.25)
        self.assertEqual(corpus_wer(["a b", "c d"], ["a b", "c x"], ident), 0.25)

    def test_primary_vs_raw_on_number_style(self):
        refs = ["Bật điều hòa hai mươi sáu độ."]
        hyps = ["Bật điều hòa 26 độ"]
        self.assertEqual(corpus_wer(refs, hyps, normalize_text), 0.0)
        self.assertGreater(corpus_wer(refs, hyps, normalize_basic), 0.0)

    def test_cer(self):
        ident = lambda s: s  # noqa: E731
        self.assertEqual(corpus_cer(["abcd"], ["abxd"], ident), 0.25)


if __name__ == "__main__":
    unittest.main()
