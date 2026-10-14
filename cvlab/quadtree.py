"""Quadtree split and region merge segmentation."""
import numpy as np


class Node:
    __slots__ = ("y", "x", "h", "w", "mean", "std", "children", "label")

    def __init__(self, y, x, h, w, mean, std):
        self.y, self.x, self.h, self.w = y, x, h, w
        self.mean, self.std = mean, std
        self.children = []
        self.label = -1


def split(img, tol=12.0, min_size=4):
    """Recursively cut a block in four until its grey levels are homogeneous (std <= tol)."""
    f = img.astype(np.float64)

    def rec(y, x, h, w):
        blk = f[y:y + h, x:x + w]
        n = Node(y, x, h, w, float(blk.mean()), float(blk.std()))
        if n.std > tol and h > min_size and w > min_size:
            h2, w2 = h // 2, w // 2
            for (yy, xx, hh, ww) in ((y, x, h2, w2), (y, x + w2, h2, w - w2),
                                     (y + h2, x, h - h2, w2), (y + h2, x + w2, h - h2, w - w2)):
                n.children.append(rec(yy, xx, hh, ww))
        return n

    return rec(0, 0, *img.shape)


def leaves(n, max_depth=None, _d=0):
    """Leaf blocks of the tree; with max_depth, blocks deeper than that are not opened."""
    if not n.children or (max_depth is not None and _d >= max_depth):
        return [n]
    return [l for c in n.children for l in leaves(c, max_depth, _d + 1)]


def depth(n):
    return 0 if not n.children else 1 + max(depth(c) for c in n.children)


def render(leaf_list, shape, use_mean=True):
    out = np.zeros(shape)
    for l in leaf_list:
        out[l.y:l.y + l.h, l.x:l.x + l.w] = l.mean
    return out


def _touch(a, b):
    ov_y = a.y < b.y + b.h and b.y < a.y + a.h
    ov_x = a.x < b.x + b.w and b.x < a.x + a.w
    return (ov_y and (a.x + a.w == b.x or b.x + b.w == a.x)) or (ov_x and (a.y + a.h == b.y or b.y + b.h == a.y))


def merge(leaf_list, tol=20.0):
    """Union adjacent leaves whose means differ by at most tol. Returns a label per leaf."""
    parent = list(range(len(leaf_list)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(leaf_list)):
        for j in range(i + 1, len(leaf_list)):
            a, b = leaf_list[i], leaf_list[j]
            if abs(a.mean - b.mean) <= tol and _touch(a, b):
                parent[find(i)] = find(j)
    roots = {}
    labels = []
    for i in range(len(leaf_list)):
        labels.append(roots.setdefault(find(i), len(roots)))
    return labels


def render_regions(leaf_list, labels, shape):
    lab = np.zeros(shape, dtype=np.int64)
    sums, cnt = {}, {}
    for l, k in zip(leaf_list, labels):
        lab[l.y:l.y + l.h, l.x:l.x + l.w] = k
        sums[k] = sums.get(k, 0.0) + l.mean * l.h * l.w
        cnt[k] = cnt.get(k, 0) + l.h * l.w
    means = {k: sums[k] / cnt[k] for k in sums}
    out = np.vectorize(means.get)(lab).astype(np.float64)
    return lab, out
