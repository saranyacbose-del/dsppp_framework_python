"""
Semantic layer and user preference learning via Inverse Reinforcement Learning.
Implements Equations 9–13 (Module 3).
"""

import math
import numpy as np
from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Semantic feature extraction
# ---------------------------------------------------------------------------

ROAD_TYPE_SCORE = {
    'motorway':    1.0,
    'trunk':       0.9,
    'primary':     0.8,
    'secondary':   0.6,
    'tertiary':    0.4,
    'residential': 0.3,
    'service':     0.2,
    'unclassified':0.1,
}


def extract_semantic_features(edge_data: Dict) -> np.ndarray:
    """
    Extract 5-dimensional semantic feature vector for an OSM road edge.
    Equation 9:  f(e) = [f_type, f_speed, f_lanes, f_surface, f_oneway]^T
    All features normalised to [0, 1].

    Parameters
    ----------
    edge_data : dict with OSM edge attributes
                (highway, maxspeed, lanes, surface, oneway)
    """
    # f_type: road type score
    highway  = edge_data.get('highway', 'unclassified')
    if isinstance(highway, list):
        highway = highway[0]
    f_type = ROAD_TYPE_SCORE.get(highway, 0.1)

    # f_speed: normalised speed limit (0–120 km/h → 0–1)
    raw_speed = edge_data.get('maxspeed', '50')
    try:
        speed = float(str(raw_speed).replace(' mph', '').replace(' km/h', '').strip())
    except (ValueError, AttributeError):
        speed = 50.0
    f_speed = min(speed / 120.0, 1.0)

    # f_lanes: normalised lane count (1–6)
    try:
        lanes = int(edge_data.get('lanes', 1))
    except (ValueError, TypeError):
        lanes = 1
    f_lanes = min(lanes / 6.0, 1.0)

    # f_surface: paved=1, unpaved=0
    surface = str(edge_data.get('surface', 'asphalt')).lower()
    f_surface = 0.0 if surface in {'unpaved', 'gravel', 'dirt', 'grass', 'sand'} else 1.0

    # f_oneway: one-way road (simpler/safer navigation)
    oneway = edge_data.get('oneway', False)
    f_oneway = 1.0 if oneway in {True, 'yes', '1'} else 0.0

    return np.array([f_type, f_speed, f_lanes, f_surface, f_oneway], dtype=float)


def build_semantic_feature_dict(graph) -> Dict:
    """
    Build {(u, v): feature_vector} dict for all edges in a NetworkX graph.
    """
    features = {}
    for u, v, data in graph.edges(data=True):
        features[(u, v)] = extract_semantic_features(data)
        features[(v, u)] = extract_semantic_features(data)   # undirected copy
    return features


def compute_semantic_weight(features: np.ndarray, alpha: np.ndarray) -> float:
    """
    Equation 10:  w_semantic(e) = sum_k( alpha_k * f_k(e) )
    """
    return float(np.dot(alpha, features))


# ---------------------------------------------------------------------------
# Inverse Reinforcement Learning (IRL) — Equations 11–13
# ---------------------------------------------------------------------------

class IRLPreferenceLearner:
    """
    Learns user preference weights alpha from observed routes
    using gradient ascent on the log-likelihood (Equations 11–13).

    Reference: Ng & Russell (2000) [5], Abbeel & Ng (2004) [6].
    """

    # Predefined preference profiles (Section 3.1, Module 3)
    PROFILES = {
        'speed':    np.array([1.0, 0.8, 0.6, 0.3, 0.2]),
        'safety':   np.array([0.2, 0.3, 0.6, 0.8, 1.0]),
        'balanced': np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
    }

    def __init__(self, n_features: int = 5,
                 learning_rate: float = 0.01,
                 max_iter: int = 500,
                 tol: float = 1e-6,
                 profile: Optional[str] = None):
        """
        Parameters
        ----------
        n_features    : dimensionality of semantic feature vector
        learning_rate : η (Equation 13)
        max_iter      : maximum gradient descent iterations
        tol           : convergence tolerance ε = 1e-6
        profile       : 'speed' | 'safety' | 'balanced' | None (random init)
        """
        self.n   = n_features
        self.eta = learning_rate
        self.max_iter = max_iter
        self.tol      = tol

        if profile and profile in self.PROFILES:
            self.alpha = self.PROFILES[profile].copy()
        else:
            self.alpha = np.ones(n_features) / n_features  # uniform init

        self.history: List[float] = []   # log-likelihood per iteration

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def fit(self, observed_routes: List[List],
            feature_dict: Dict,
            random_routes_per_obs: int = 5,
            graph=None) -> np.ndarray:
        """
        Learn alpha from observed routes R = {r_1, ..., r_m}.

        Parameters
        ----------
        observed_routes       : list of paths (each path = list of nodes)
        feature_dict          : {(u,v): feature_vector}
        random_routes_per_obs : number of random comparison routes per observed route
        graph                 : NetworkX graph (for sampling random routes)

        Returns
        -------
        alpha : learned preference weight vector
        """
        for iteration in range(self.max_iter):
            grad   = np.zeros(self.n)
            log_ll = 0.0

            for obs_route in observed_routes:
                c_obs = self._route_cost(obs_route, feature_dict)

                # Sample random comparison routes
                comparison_routes = self._sample_comparisons(
                    obs_route, random_routes_per_obs, graph
                )

                for rand_route in comparison_routes:
                    c_rand = self._route_cost(rand_route, feature_dict)

                    # Log-likelihood gradient (Equation 12 approximation)
                    delta_cost = c_rand - c_obs
                    log_ll    += delta_cost

                    # Feature expectation difference
                    f_obs  = self._route_features(obs_route,  feature_dict)
                    f_rand = self._route_features(rand_route, feature_dict)
                    grad  += f_obs - f_rand

            # Normalise gradient
            if len(observed_routes) > 0:
                grad   /= len(observed_routes)
                log_ll /= len(observed_routes)

            # Gradient ascent (Equation 13)
            alpha_new = self.alpha + self.eta * grad

            # Project to non-negative simplex
            alpha_new = np.clip(alpha_new, 0.0, None)
            s = alpha_new.sum()
            if s > 0:
                alpha_new /= s

            # Convergence check
            update_mag = np.linalg.norm(alpha_new - self.alpha)
            self.alpha = alpha_new
            self.history.append(log_ll)

            if update_mag < self.tol:
                break

        return self.alpha

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _route_cost(self, route: List, feature_dict: Dict) -> float:
        """Total semantic cost of a route given current alpha."""
        cost = 0.0
        for i in range(len(route) - 1):
            edge = (route[i], route[i+1])
            feat = feature_dict.get(edge, np.zeros(self.n))
            cost += 1.0 - float(np.dot(self.alpha, feat))   # Eq. 19 (w_base=1)
        return cost

    def _route_features(self, route: List, feature_dict: Dict) -> np.ndarray:
        """Average feature vector across all edges of a route."""
        if len(route) < 2:
            return np.zeros(self.n)
        total = np.zeros(self.n)
        for i in range(len(route) - 1):
            edge = (route[i], route[i+1])
            total += feature_dict.get(edge, np.zeros(self.n))
        return total / (len(route) - 1)

    def _sample_comparisons(self, obs_route: List,
                             n_samples: int, graph) -> List[List]:
        """Generate random alternative routes for IRL comparison."""
        if graph is None or len(obs_route) < 2:
            return []
        import random
        nodes = list(graph.nodes())
        comparisons = []
        start, goal  = obs_route[0], obs_route[-1]
        for _ in range(n_samples):
            # Random walk of same length
            path = [start]
            cur  = start
            for _ in range(len(obs_route) - 1):
                nbrs = list(graph.neighbors(cur))
                if not nbrs:
                    break
                cur = random.choice(nbrs)
                path.append(cur)
            comparisons.append(path)
        return comparisons

    def predict_profile(self) -> str:
        """Return closest named profile to current alpha."""
        best, best_d = 'balanced', float('inf')
        for name, vec in self.PROFILES.items():
            d = np.linalg.norm(self.alpha - vec)
            if d < best_d:
                best, best_d = name, d
        return best
