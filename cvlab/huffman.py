"""Shannon entropy and Huffman coding."""
import heapq
import itertools
import numpy as np


def entropy(values):
    v, c = np.unique(np.asarray(values).ravel(), return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def build_tree(freqs):
    """freqs: dict symbol -> count. Returns nested tuples (leaf: symbol, node: (left, right))."""
    tie = itertools.count()
    heap = [(c, next(tie), s) for s, c in freqs.items()]
    heapq.heapify(heap)
    if len(heap) == 1:
        return (heap[0][2], None)
    while len(heap) > 1:
        c1, _, a = heapq.heappop(heap)
        c2, _, b = heapq.heappop(heap)
        heapq.heappush(heap, (c1 + c2, next(tie), (a, b)))
    return heap[0][2]


def codes(tree, prefix=""):
    if not isinstance(tree, tuple):
        return {tree: prefix or "0"}
    l, r = tree
    out = {}
    if l is not None:
        out.update(codes(l, prefix + "0"))
    if r is not None:
        out.update(codes(r, prefix + "1"))
    return out


def encode(values):
    v, c = np.unique(np.asarray(values).ravel(), return_counts=True)
    freqs = {int(a): int(b) for a, b in zip(v, c)}
    table = codes(build_tree(freqs))
    return table, freqs


def avg_length(table, freqs):
    n = sum(freqs.values())
    return sum(len(table[s]) * c for s, c in freqs.items()) / n


def encode_bits(values, table):
    return "".join(table[int(x)] for x in np.asarray(values).ravel())


def decode_bits(bits, table):
    inv = {c: s for s, c in table.items()}
    out, cur = [], ""
    for b in bits:
        cur += b
        if cur in inv:
            out.append(inv[cur])
            cur = ""
    return out
