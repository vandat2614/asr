"""Text normalization used to score ASR output (WER/CER). See ADR-0003.

Applied to BOTH the reference and the hypothesis. The raw ASR text is kept
untouched for the downstream LLM; this module is for scoring only.

Steps of `normalize_text`:
  1. Unicode NFC, lowercase.
  2. Digits to Vietnamese words (times, thousands separators, decimals,
     percent, phone-like numbers).
  3. Merge spelling variants: ngan -> nghin, lam -> nam, mot (mốt) -> mot,
     le -> linh, "muoi tu" -> "muoi bon".
  4. Replace punctuation with spaces and collapse whitespace.

`normalize_basic` does steps 1 and 4 only (no digit expansion, no variant
merge). It is used for the secondary "raw" WER.
"""
import re
import unicodedata

_DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
_GROUPS = ["", "nghìn", "triệu", "tỷ"]
_MAX_READABLE_DIGITS = 12


def _read_triple(n, full):
    """Read 0 <= n <= 999. `full` forces 'không trăm' when the hundreds are 0."""
    h, t, u = n // 100, (n // 10) % 10, n % 10
    words = []
    if h > 0 or full:
        words += [_DIGITS[h], "trăm"]
    if t == 0:
        if u > 0:
            if h > 0 or full:
                words.append("linh")
            words.append(_DIGITS[u])
    elif t == 1:
        words.append("mười")
        if u > 0:
            words.append(_DIGITS[u])
    else:
        words += [_DIGITS[t], "mươi"]
        if u > 0:
            words.append(_DIGITS[u])
    return words


def read_int(n):
    """Read a non-negative integer (< 10**12) as canonical Vietnamese words."""
    if n == 0:
        return _DIGITS[0]
    triples = []
    while n > 0:
        triples.append(n % 1000)
        n //= 1000
    words = []
    for i in range(len(triples) - 1, -1, -1):
        value = triples[i]
        if value == 0:
            continue
        is_top = i == len(triples) - 1
        words += _read_triple(value, full=not is_top)
        if _GROUPS[i]:
            words.append(_GROUPS[i])
    return " ".join(words)


def _spell_digits(s):
    return " ".join(_DIGITS[int(c)] for c in s)


def _read_number_token(s):
    """Phone numbers, leading zeros and very long tokens are read digit by digit."""
    if (len(s) > 1 and s.startswith("0")) or len(s) > _MAX_READABLE_DIGITS:
        return _spell_digits(s)
    return read_int(int(s))


_TIME_H = re.compile(r"(?<!\d)(\d{1,2})\s?h(?:(\d{2}))?(?!\w)")
_TIME_COLON = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)")
_THOUSANDS = re.compile(r"(?<!\d)\d{1,3}(?:[.,]\d{3})+(?!\d)")
_DECIMAL = re.compile(r"(?<!\d)(\d+)[.,](\d+)(?!\d)")
_NUMBER = re.compile(r"\d+")


def _expand_digits(text):
    text = _TIME_H.sub(lambda m: f" {m.group(1)} giờ {m.group(2) or ''} ", text)
    text = _TIME_COLON.sub(lambda m: f" {m.group(1)} giờ {m.group(2)} ", text)
    text = _THOUSANDS.sub(lambda m: re.sub(r"[.,]", "", m.group(0)), text)
    text = _DECIMAL.sub(
        lambda m: f" {_read_number_token(m.group(1))} phẩy {_spell_digits(m.group(2))} ",
        text,
    )
    text = text.replace("%", " phần trăm ")
    return _NUMBER.sub(lambda m: f" {_read_number_token(m.group(0))} ", text)


_VARIANTS = [
    (re.compile(r"\bngàn\b"), "nghìn"),
    (re.compile(r"\blăm\b"), "năm"),
    (re.compile(r"\bmốt\b"), "một"),
    (re.compile(r"\blẻ\b"), "linh"),
    (re.compile(r"\b(mươi|mười)\s+tư\b"), r"\1 bốn"),
]


def _merge_variants(text):
    for rx, repl in _VARIANTS:
        text = rx.sub(repl, text)
    return text


def _clean(text):
    text = re.sub(r"[^\w\s]|_", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_basic(text):
    """NFC, lowercase, punctuation removal. No digit expansion."""
    return _clean(unicodedata.normalize("NFC", text).lower())


def normalize_text(text):
    """Full normalization used for the primary WER."""
    text = unicodedata.normalize("NFC", text).lower()
    text = _expand_digits(text)
    text = _merge_variants(text)
    return _clean(text)
