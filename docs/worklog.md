# Worklog

## 2026-11-01
- Created repo, docs skeleton and ADR template.
- ADR-0001: development compute (Kaggle T4 x2).
- Downloaded and extracted the data package into data/raw/ (1.2 GB).
- Removed __MACOSX junk. Wrote scripts/data_stats.py.
- Data check: train 1545 samples / 5036 utterances / 6.83 h; public_test 634 samples / 1933 utterances / 2.76 h. No missing or unreadable audio.
- Wrote docs/data-overview.md.
- Wrote scripts/make_split.py (grouped split by near-duplicate final transcripts).
- Primary split (threshold 0.9): 1236 train / 309 val samples, 1477 groups, 98 samples in multi-sample groups.
- Strict split (threshold 0.8): 1236 train / 309 val samples, 1304 groups, 285 samples in multi-sample groups.
- ADR-0002: use 0.9 as primary and 0.8 as a stress test.
- Wrote scripts/transcript_stats.py and scripts/number_style_stats.py (aggregate counts only, no transcript text).
- Findings: labels mostly spell numbers out; times appear as "D gio" or word form (no 8:30 form); money uses nghin/ngan; D.DDD only in assistant text.
- Wrote src/asr/normalize.py with unit tests (10 passing).
- ADR-0003: primary WER on normalized text, secondary raw WER.
