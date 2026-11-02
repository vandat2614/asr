"""Tests for tone-placement unification. Run: python3 -m unittest discover -s tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asr.normalize import normalize_basic, normalize_text  # noqa: E402


class TonePlacementTest(unittest.TestCase):
    def test_old_and_new_style_match(self):
        pairs = [
            ("hoá", "hóa"),
            ("khoẻ", "khỏe"),
            ("thuý", "thúy"),
            ("huỷ", "hủy"),
            ("khoá", "khóa"),
        ]
        for old, new in pairs:
            self.assertEqual(normalize_text(old), normalize_text(new), (old, new))

    def test_inside_a_sentence(self):
        self.assertEqual(
            normalize_text("Huỷ lịch, khoá cửa."),
            normalize_text("Hủy lịch, khóa cửa."),
        )

    def test_closed_syllables_untouched(self):
        for word in ["hoàn", "toán", "khoản"]:
            self.assertEqual(normalize_text(word), word)

    def test_qu_syllables_untouched(self):
        for word in ["quý", "quỳ", "quả"]:
            self.assertEqual(normalize_text(word), word)

    def test_word_continuing_after_uy_untouched(self):
        self.assertEqual(normalize_text("khuỷu"), "khuỷu")

    def test_basic_normalization_keeps_raw_tone_style(self):
        self.assertEqual(normalize_basic("Hoá"), "hoá")
        self.assertNotEqual(normalize_basic("hoá"), normalize_basic("hóa"))


if __name__ == "__main__":
    unittest.main()
