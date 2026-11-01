"""Tests for align_ops. Run: python3 -m unittest discover -s tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asr.metrics import align_ops, edit_distance  # noqa: E402


class AlignTest(unittest.TestCase):
    def test_identical(self):
        ops = align_ops(["a", "b"], ["a", "b"])
        self.assertEqual([o[0] for o in ops], ["ok", "ok"])

    def test_substitution(self):
        ops = align_ops(["a", "b", "c"], ["a", "x", "c"])
        self.assertEqual(ops, [("ok", "a", "a"), ("sub", "b", "x"), ("ok", "c", "c")])

    def test_deletion_and_insertion(self):
        self.assertEqual(
            align_ops(["a", "b"], ["a"]), [("ok", "a", "a"), ("del", "b", None)]
        )
        self.assertEqual(
            align_ops(["a"], ["a", "b"]), [("ok", "a", "a"), ("ins", None, "b")]
        )

    def test_empty_sides(self):
        self.assertEqual(align_ops([], ["a"]), [("ins", None, "a")])
        self.assertEqual(align_ops(["a"], []), [("del", "a", None)])
        self.assertEqual(align_ops([], []), [])

    def test_error_count_matches_edit_distance(self):
        ref = "bật đèn phòng khách lúc tám giờ".split()
        hyp = "bật đèn khách phòng tám rồi".split()
        errors = sum(1 for op, _, _ in align_ops(ref, hyp) if op != "ok")
        self.assertEqual(errors, edit_distance(ref, hyp))


if __name__ == "__main__":
    unittest.main()
