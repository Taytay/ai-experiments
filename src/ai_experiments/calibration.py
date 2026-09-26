"""Calibration and selective-accuracy metrics over per-option scores (PLAN step 49, QUESTIONS.md STAT-4; reused by rows 52 and 54).

Numpy only. The temperature fit follows jqv's `TemperatureScaler.fit` (a log-spaced grid over T in [0.05, 100], then golden-section
refinement; an unbounded optimiser diverges on near-separable sets, where the NLL keeps falling as T -> 0), and the metric
definitions follow kev's `metrics.py` (ECE over ten equal-width confidence bins; the risk-coverage curve over thresholds on the
top-option probability with ties accepted together; AURC as the coverage-weighted mean risk; a threshold chosen as the lowest
confidence whose accepted set keeps the error rate within budget). Both repositories sit beside this one (`../jqv`, `../kev`).

A row's scores are one vector per item (any length: REAL-6 users have 8 to 20 categories); probabilities are softmax(z / T).
"""
import math

import numpy as np


def softmax(z, t=1.0):
    z = np.asarray(z, float) / t
    e = np.exp(z - z.max())
    return e / e.sum()


def pad(scores):
    """Ragged score vectors as one (N, K_max) matrix, -inf where a user has fewer categories."""
    k = max(len(z) for z in scores); m = np.full((len(scores), k), -np.inf)
    for i, z in enumerate(scores):
        m[i, :len(z)] = z
    return m


def nll(scores, labels, t=1.0):
    z = scores if isinstance(scores, np.ndarray) else pad(scores)
    z = z / t; mx = z.max(1, keepdims=True)
    lse = mx[:, 0] + np.log(np.exp(z - mx).sum(1))
    return float(np.mean(lse - z[np.arange(len(z)), np.asarray(labels)]))


def fit_temperature(scores, labels, t_min=0.05, t_max=100.0, grid=200, refine=40):
    """T minimising the NLL of softmax(z / T), by a bounded 1-D search (grid in log T, then golden section)."""
    z = pad(scores)
    f = lambda lt: nll(z, labels, math.exp(lt))  # noqa: E731
    pts = np.linspace(math.log(t_min), math.log(t_max), grid)
    vals = [f(p) for p in pts]
    i = int(np.argmin(vals))
    a, b = pts[max(i - 1, 0)], pts[min(i + 1, grid - 1)]
    phi = (math.sqrt(5) - 1) / 2
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(refine):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a); fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a); fd = f(d)
    return float(math.exp((a + b) / 2))


def summarise(probs, labels):
    """Per item: confidence (top probability), correct (top option is gold), Brier (squared error over all options), NLL."""
    conf, corr, brier, ll = [], [], [], []
    for p, y in zip(probs, labels):
        p = np.asarray(p, float); k = int(np.argmax(p))
        onehot = np.zeros_like(p); onehot[y] = 1
        conf.append(p[k]); corr.append(k == y); brier.append(float(((p - onehot) ** 2).sum())); ll.append(-math.log(max(p[y], 1e-300)))
    return np.array(conf), np.array(corr, float), np.array(brier), np.array(ll)


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi) if hi < 1 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def _curve(conf, correct):
    """Thresholds (distinct confidences, high to low) with the accepted count and error count at each."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    order = np.argsort(-conf, kind="stable"); c, w = conf[order], 1 - correct[order]
    last = np.r_[np.flatnonzero(np.diff(c) != 0), len(c) - 1]  # accept ties together
    return c[last], last + 1, np.cumsum(w)[last]


def aurc(conf, correct):
    _, acc, err = _curve(conf, correct)
    return float(np.sum(np.diff(np.r_[0, acc]) * err / acc) / acc[-1]) if len(acc) else 0.0


def select_threshold(conf, correct, budget):
    """The lowest confidence threshold whose accepted set has error rate <= budget (None when no threshold does)."""
    th, acc, err = _curve(conf, correct)
    ok = np.flatnonzero(err <= budget * acc)
    return float(th[ok[-1]]) if len(ok) else None


def coverage_at_precision(conf, correct, precision):
    """Largest share of items that can be accepted with accuracy >= precision, the threshold chosen on the same items (oracle)."""
    t = select_threshold(conf, correct, 1 - precision)
    return 0.0 if t is None else float(np.mean(np.asarray(conf) >= t))
