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


def align_ops(ref, hyp):
    """Align two token lists. Returns a list of (op, ref_token, hyp_token).

    op is one of 'ok', 'sub', 'del', 'ins'. The number of non-'ok' ops equals
    edit_distance(ref, hyp).
    """
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]),
            )
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("ok" if ref[i - 1] == hyp[j - 1] else "sub", ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1]))
            j -= 1
    ops.reverse()
    return ops
