"""Tests for the spoken short form of tens + units. Run: python3 -m unittest discover -s tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asr.normalize import normalize_text  # noqa: E402


class SpokenShortFormTest(unittest.TestCase):
    def test_optional_muoi_between_tens_and_units(self):
        self.assertEqual(normalize_text("bốn lăm phút"), normalize_text("45 phút"))
        self.assertEqual(normalize_text("xăng RON chín lăm"), normalize_text("xăng RON95"))
        self.assertEqual(normalize_text("hai mươi lăm độ"), normalize_text("hai lăm độ"))

    def test_bare_tens_keep_muoi(self):
        self.assertEqual(normalize_text("30 phần trăm"), "ba mươi phần trăm")
        self.assertEqual(normalize_text("Chuyển 50 nghìn"), "chuyển năm mươi nghìn")


if __name__ == "__main__":
    unittest.main()
