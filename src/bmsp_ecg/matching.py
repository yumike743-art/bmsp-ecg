"""Exact ordered QRS matching with a sparse dynamic program."""

from bisect import bisect_left, bisect_right


def match_qrs(reference, detected, tolerance):
    """Maximize matched count, then minimize summed absolute sample error.

    For sorted sample locations a crossing pair can be uncrossed without losing
    feasibility or raising total absolute error. A Fenwick tree stores the
    best monotone pairing ending in each prefix of detector indices.
    """
    if tolerance < 0 or any(a > b for a, b in zip(reference, reference[1:])) or any(
        a > b for a, b in zip(detected, detected[1:])
    ):
        raise ValueError("Matching requires nonnegative tolerance and sorted samples")
    size = len(detected)
    tree = [(0, 0, 0)] * (size + 1)
    chain = [None]
    for ref_index, time in enumerate(reference):
        left = bisect_left(detected, time - tolerance)
        right = bisect_right(detected, time + tolerance)
        batch = []
        for det_index in range(left, right):
            cursor = det_index
            best = (0, 0, 0)
            while cursor:
                best = max(best, tree[cursor])
                cursor -= cursor & -cursor
            count, neg_error, neg_previous = best
            chain.append((ref_index, det_index, -neg_previous))
            score = (count + 1, neg_error - abs(time - detected[det_index]), -(len(chain) - 1))
            batch.append((det_index + 1, score))
        for cursor, score in batch:
            while cursor <= size:
                tree[cursor] = max(tree[cursor], score)
                cursor += cursor & -cursor
    cursor = size
    best = (0, 0, 0)
    while cursor:
        best = max(best, tree[cursor])
        cursor -= cursor & -cursor
    pairs = []
    pointer = -best[2]
    while pointer:
        ref_index, det_index, pointer = chain[pointer]
        pairs.append((ref_index, det_index))
    return pairs[::-1]
