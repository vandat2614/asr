"""Tests for src/asr/normalize.py. Run: python3 -m unittest discover -s tests"""
import sys
import unicodedata
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asr.normalize import normalize_basic, normalize_text, read_int  # noqa: E402


class ReadIntTest(unittest.TestCase):
    def test_cases(self):
        cases = {
            0: "không",
            5: "năm",
            10: "mười",
            11: "mười một",
            15: "mười năm",
            21: "hai mươi một",
            25: "hai mươi năm",
            26: "hai mươi sáu",
            100: "một trăm",
            105: "một trăm linh năm",
            110: "một trăm mười",
            500: "năm trăm",
            1000: "một nghìn",
            1005: "một nghìn không trăm linh năm",
            1500: "một nghìn năm trăm",
            500000: "năm trăm nghìn",
            1500000: "một triệu năm trăm nghìn",
            2000000000: "hai tỷ",
        }
        for n, expected in cases.items():
            self.assertEqual(read_int(n), expected, n)


class NormalizeTest(unittest.TestCase):
    def check(self, raw, expected):
        self.assertEqual(normalize_text(raw), expected, raw)

    def test_case_and_punctuation(self):
        self.check("Bật đèn phòng khách, nhé!", "bật đèn phòng khách nhé")
        self.check("  Mấy   giờ rồi?  ", "mấy giờ rồi")

    def test_unicode_nfd_equals_nfc(self):
        raw = "Tắt đèn"
        nfd = unicodedata.normalize("NFD", raw)
        self.assertEqual(normalize_text(nfd), normalize_text(raw))

    def test_temperature_and_time(self):
        self.check("Bật điều hòa 26 độ.", "bật điều hòa hai mươi sáu độ")
        self.check("Đặt báo thức 8 giờ", "đặt báo thức tám giờ")
        self.check("Đặt báo thức 8h30", "đặt báo thức tám giờ ba mươi")
        self.check("Họp lúc 8:30", "họp lúc tám giờ ba mươi")
        self.check("Họp lúc 8:05", "họp lúc tám giờ không năm")
        self.check("Mua 8 hộp sữa", "mua tám hộp sữa")
        self.check("7 giờ kém 10", "bảy giờ kém mười")

    def test_money(self):
        self.check("Chuyển 500.000 đồng", "chuyển năm trăm nghìn đồng")
        self.check("Chuyển 500 ngàn", "chuyển năm trăm nghìn")
        self.check("Chuyển 1.500.000", "chuyển một triệu năm trăm nghìn")
        self.check("Chuyển 2 triệu", "chuyển hai triệu")

    def test_spelling_variants_merge(self):
        self.assertEqual(normalize_text("hai mươi lăm"), normalize_text("hai mươi năm"))
        self.assertEqual(normalize_text("hai mươi mốt"), normalize_text("hai mươi một"))
        self.assertEqual(normalize_text("một trăm lẻ năm"), normalize_text("105"))
        self.assertEqual(normalize_text("hai mươi tư độ"), "hai mươi bốn độ")
        self.assertEqual(normalize_text("25 độ"), normalize_text("hai mươi lăm độ"))

    def test_tu_is_kept_outside_numbers(self):
        self.check("Thứ tư tuần sau", "thứ tư tuần sau")

    def test_phone_and_decimal_and_percent(self):
        self.check("Gọi 0901234567", "gọi không chín không một hai ba bốn năm sáu bảy")
        self.check("Pin còn 1,5", "pin còn một phẩy năm")
        self.check("Giảm 50%", "giảm năm mươi phần trăm")

    def test_digits_and_words_agree(self):
        self.assertEqual(normalize_text("Đặt xe lúc 8 giờ"), normalize_text("đặt xe lúc tám giờ"))

    def test_basic_keeps_digits(self):
        self.assertEqual(normalize_basic("Bật 26 độ."), "bật 26 độ")


if __name__ == "__main__":
    unittest.main()
