"""
Evaluation metrics including Personalized Path Efficiency (PPE).
Equations 29–32 from the paper.
"""

import math
import time
import numpy as np
from typing import List, Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# PPE metric (Equations 29–32)
# ---------------------------------------------------------------------------

def compute_ppe(path: List,
                node_pos_fn,
                alpha: np.ndarray,
                feature_dict: Dict,
                obstacles: List[Dict],
                computation_time_ms: float,
                w_length:   float = 0.4,
                w_semantic: float = 0.3,
                w_safety:   float = 0.3,
                d_threshold: float = 2.0) -> float:
    """
    Personalized Path Efficiency (PPE) — Equation 29.

    PPE = (W_length * R_length + W_semantic * R_semantic + W_safety * R_safety)
          / T_computation

    Parameters
    ----------
    path               : list of nodes
    node_pos_fn        : callable(node) -> (x, y)
    alpha              : user preference weights
    feature_dict       : {(u,v): feature_vector}
    obstacles          : list of obstacle dicts with 'px', 'py'
    computation_time_ms: planning time in milliseconds
    """
    if len(path) < 2 or computation_time_ms <= 0:
        return 0.0

    r_length   = _length_ratio(path, node_pos_fn)
    r_semantic = _semantic_score(path, alpha, feature_dict)
    r_safety   = _safety_ratio(path, node_pos_fn, obstacles, d_threshold)

    numerator = (w_length * r_length
                 + w_semantic * r_semantic
                 + w_safety * r_safety)
    return numerator / computation_time_ms


def _length_ratio(path: List, node_pos_fn) -> float:
    """
    Equation 30: R_length = L_optimal / L_actual
    L_optimal = straight-line start-to-goal distance.
    """
    if len(path) < 2:
        return 0.0
    xs, ys = node_pos_fn(path[0])
    xg, yg = node_pos_fn(path[-1])
    l_optimal = math.hypot(xg - xs, yg - ys)

    l_actual = sum(
        math.hypot(*(np.array(node_pos_fn(path[i+1]))
                     - np.array(node_pos_fn(path[i]))))
        for i in range(len(path) - 1)
    )
    return l_optimal / l_actual if l_actual > 0 else 0.0


def _semantic_score(path: List, alpha: np.ndarray,
                    feature_dict: Dict) -> float:
    """
    Equation 31: R_semantic = (1/|P|) * sum_e sum_k alpha_k * s_k(e)
    """
    if len(path) < 2:
        return 0.0
    n = len(alpha)
    total = 0.0
    for i in range(len(path) - 1):
        edge = (path[i], path[i+1])
        feat = feature_dict.get(edge, np.zeros(n))
        total += float(np.dot(alpha, feat))
    return total / len(path)


def _safety_ratio(path: List, node_pos_fn,
                  obstacles: List[Dict],
                  d_threshold: float = 2.0) -> float:
    """
    Equation 32: R_safety = min_{n in P, o in O} d(n, o) / d_threshold
    Values > 1.0 indicate safe paths with sufficient clearance.
    """
    if not obstacles or len(path) < 1:
        return 1.0

    min_dist = float('inf')
    for node in path:
        x, y = node_pos_fn(node)
        for obs in obstacles:
            d = math.hypot(obs['px'] - x, obs['py'] - y)
            min_dist = min(min_dist, d)

    return min_dist / d_threshold if d_threshold > 0 else 0.0


# ---------------------------------------------------------------------------
# Path metrics
# ---------------------------------------------------------------------------

def path_length(path: List, node_pos_fn) -> float:
    """Total Euclidean path length in world units."""
    total = 0.0
    for i in range(len(path) - 1):
        x0, y0 = node_pos_fn(path[i])
        x1, y1 = node_pos_fn(path[i+1])
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def path_smoothness(path: List, node_pos_fn) -> float:
    """
    Average angular deviation per segment (degrees).
    Lower is smoother (Table II metric).
    """
    if len(path) < 3:
        return 0.0
    angles = []
    for i in range(1, len(path) - 1):
        x0, y0 = node_pos_fn(path[i-1])
        x1, y1 = node_pos_fn(path[i])
        x2, y2 = node_pos_fn(path[i+1])

        v1 = np.array([x1-x0, y1-y0])
        v2 = np.array([x2-x1, y2-y1])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angles.append(math.degrees(math.acos(cos_a)))
    return float(np.mean(angles)) if angles else 0.0


def safety_margin(path: List, node_pos_fn, obstacles: List[Dict]) -> float:
    """Minimum obstacle clearance along the path (world units)."""
    if not obstacles or not path:
        return float('inf')
    min_d = float('inf')
    for node in path:
        x, y = node_pos_fn(node)
        for obs in obstacles:
            d = math.hypot(obs['px'] - x, obs['py'] - y)
            min_d = min(min_d, d)
    return min_d


# ---------------------------------------------------------------------------
# Statistical analysis helpers (Section 4.7)
# ---------------------------------------------------------------------------

def bootstrap_ci(data: np.ndarray,
                 n_iter: int = 10000,
                 confidence: float = 0.95) -> Tuple[float, float]:
    """
    Bootstrap confidence interval.
    Section 4.7: 95% CI with 10,000 iterations.
    """
    rng = np.random.default_rng()
    means = [rng.choice(data, size=len(data), replace=True).mean()
             for _ in range(n_iter)]
    lo = (1 - confidence) / 2
    hi = 1 - lo
    return float(np.quantile(means, lo)), float(np.quantile(means, hi))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cohen's d effect size.  Equation in Section 4.7.
    d = (mu_1 - mu_2) / s_pooled
    """
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    s_pooled = math.sqrt(
        ((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1))
        / (n1 + n2 - 2)
    )
    return float((np.mean(a) - np.mean(b)) / s_pooled) if s_pooled > 0 else 0.0


def one_way_anova(*groups: np.ndarray) -> Tuple[float, float]:
    """One-way ANOVA.  Returns (F-statistic, p-value)."""
    from scipy import stats
    return stats.f_oneway(*groups)


def effect_size_label(d: float) -> str:
    """Cohen (1988) thresholds for effect size classification."""
    d = abs(d)
    if d < 0.5:  return 'small'
    if d < 1.5:  return 'medium'
    if d < 3.0:  return 'large'
    return 'very large'
