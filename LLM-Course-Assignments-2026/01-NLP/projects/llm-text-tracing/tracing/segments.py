"""Four Qwen windows distributed across a text, with at most 64 tokens in total."""
import numpy as np

COUNT = 4
WIDTH = 16
MEAN = "qwen-segment-mean"
CHANGE = "qwen-segment-change"
METHODS = (MEAN, CHANGE)


def segment_ranges(length):
    """Return four ordered half-open token ranges. Empty ranges come last."""
    if length < 1:
        raise ValueError("分词后文本为空")
    if length >= COUNT * WIDTH:
        starts = [i * (length - WIDTH) // (COUNT - 1) for i in range(COUNT)]
        return [(start, start + WIDTH) for start in starts]
    width, extra = divmod(length, COUNT)
    edges = [i * width + min(i, extra) for i in range(COUNT + 1)]
    return list(zip(edges[:-1], edges[1:]))


def unit(values):
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def segment_vectors(windows, method):
    """Empty slots are zero. L2 normalization makes sums equivalent to means here."""
    slots = unit(windows)
    mean = unit(slots.sum(axis=1))
    if method == MEAN:
        return mean
    if method != CHANGE:
        raise ValueError(f"未知分段方法 {method}")
    valid = np.any(windows != 0, axis=-1)
    pairs = (valid[:, 1:] & valid[:, :-1])[..., None]
    change = unit((np.abs(np.diff(slots, axis=1)) * pairs).sum(axis=1))
    return unit(np.concatenate([mean, change], axis=1))


def adjacent_distances(windows):
    """Cosine distances for the UI, one per pair of populated adjacent slots."""
    slots = unit(windows[np.any(windows != 0, axis=-1)])
    return np.clip(1 - (slots[:-1] * slots[1:]).sum(axis=-1), 0, 2).tolist()
