"""WER / CER with a pluggable text normalizer (see ADR-0003)."""


def edit_distance(ref, hyp):
    """Levenshtein distance between two sequences."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def corpus_wer(refs, hyps, normalize):
    """Corpus-level WER: total word errors / total reference words."""
    errors = words = 0
    for ref, hyp in zip(refs, hyps):
        ref_words = normalize(ref).split()
        hyp_words = normalize(hyp).split()
        errors += edit_distance(ref_words, hyp_words)
        words += len(ref_words)
    return errors / max(words, 1)


def corpus_cer(refs, hyps, normalize):
    """Corpus-level CER on normalized text (spaces included)."""
    errors = chars = 0
    for ref, hyp in zip(refs, hyps):
        ref_chars = normalize(ref)
        errors += edit_distance(ref_chars, normalize(hyp))
        chars += len(ref_chars)
    return errors / max(chars, 1)
