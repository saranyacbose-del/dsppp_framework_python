"""
DSPPP: Dynamic Semantic Personalized Path Planning
Improved A* with dynamic penalty costs, semantic awareness, and personalized routing.
"""

import heapq
import time
import math
import numpy as np
from typing import List, Tuple, Dict, Optional, Set


class DSPPPPlanner:
    """
    Improved A* path planner with:
      - Dynamic penalty costs (Kalman-predicted obstacles)
      - Semantic cost layer (OSM road attributes + IRL preferences)
      - Path smoothness cost
      - Real-time replanning (local / global / smoothing)
    """

    # Default parameters (Table I in paper)
    LAMBDA_BASE   = 5.0    # Base obstacle penalty weight
    SIGMA         = 2.0    # Penalty spatial spread (grid units)
    DELTA_T       = 1.0    # Planning lookahead time (s)
    GAMMA         = 0.15   # Heuristic density sensitivity
    BETA          = 0.1    # Path smoothness weight
    D_SAFETY      = 2.0    # Safety margin (grid units)
    R_SENSE       = 5.0    # Obstacle sensing radius
    T_MAX         = 5.0    # Max planning time (s)
    W_BASE        = 1.0    # Baseline semantic cost
    V_MAX         = 10.0   # Max obstacle speed (m/s)

    def __init__(self, graph, penalty_map: Dict, alpha: np.ndarray,
                 params: Optional[Dict] = None):
        """
        Parameters
        ----------
        graph       : NetworkX graph or grid adjacency dict  {node: [neighbors]}
        penalty_map : {node: float}  dynamic penalty values (updated externally at 10 Hz)
        alpha       : np.ndarray     user preference weights (from IRL module)
        params      : optional dict to override default parameters
        """
        self.graph       = graph
        self.penalty_map = penalty_map
        self.alpha       = alpha

        p = params or {}
        self.lambda_base = p.get('lambda_base', self.LAMBDA_BASE)
        self.sigma       = p.get('sigma',       self.SIGMA)
        self.delta_t     = p.get('delta_t',     self.DELTA_T)
        self.gamma       = p.get('gamma',       self.GAMMA)
        self.beta        = p.get('beta',        self.BETA)
        self.d_safety    = p.get('d_safety',    self.D_SAFETY)
        self.r_sense     = p.get('r_sense',     self.R_SENSE)
        self.t_max       = p.get('t_max',       self.T_MAX)
        self.w_base      = p.get('w_base',      self.W_BASE)
        self.v_max       = p.get('v_max',       self.V_MAX)

        # Semantic feature dict: {edge: np.ndarray of features}
        self.semantic_features: Dict = {}

        # Current path (for replanning)
        self.current_path: List = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, start, goal, obstacles: List[Dict]) -> Tuple[List, float]:
        """
        Run improved A* from start to goal.

        Returns
        -------
        path : list of nodes
        cost : total path cost
        """
        t_start = time.perf_counter()
        path = self._astar(start, goal, obstacles, t_start)
        self.current_path = path
        cost = self._path_cost(path, obstacles)
        return path, cost

    def replan(self, current_pos, goal, obstacles: List[Dict],
               current_cost: float, initial_cost: float) -> Tuple[List, str]:
        """
        Adaptive replanning.  Selects local / global / smoothing strategy.

        Returns
        -------
        new_path : list of nodes
        mode     : 'local' | 'global' | 'smooth'
        """
        mode = self._select_replan_mode(current_pos, obstacles, current_cost, initial_cost)
        t_start = time.perf_counter()

        if mode == 'local':
            k = 5
            lookahead = self._astar(current_pos, goal, obstacles, t_start,
                                    max_nodes=k)
            # Splice: use lookahead then continue old path
            try:
                splice_idx = self.current_path.index(lookahead[-1]) if lookahead else 0
                new_path = lookahead + self.current_path[splice_idx + 1:]
            except ValueError:
                new_path = lookahead
        elif mode == 'global':
            new_path = self._astar(current_pos, goal, obstacles, t_start)
        else:  # smooth
            new_path = self._smooth_path(self.current_path)

        self.current_path = new_path
        return new_path, mode

    # ------------------------------------------------------------------
    # Core A* (Algorithm 1)
    # ------------------------------------------------------------------

    def _astar(self, start, goal, obstacles, t_start,
               max_nodes: Optional[int] = None) -> List:
        open_heap: List[Tuple[float, any]] = []
        g_score: Dict = {start: 0.0}
        f_score: Dict = {start: self._heuristic(start, goal, obstacles)}
        parent: Dict  = {start: None}
        closed: Set   = set()

        heapq.heappush(open_heap, (f_score[start], start))
        nodes_expanded = 0

        while open_heap:
            # Time budget check
            if time.perf_counter() - t_start > self.t_max:
                return self._best_partial_path(parent, g_score, goal)

            _, current = heapq.heappop(open_heap)

            if current == goal:
                return self._reconstruct_path(parent, current)

            if current in closed:
                continue
            closed.add(current)
            nodes_expanded += 1

            if max_nodes and nodes_expanded >= max_nodes:
                return self._reconstruct_path(parent, current)

            for neighbor in self._neighbors(current):
                if neighbor in closed:
                    continue

                tentative_g = (g_score[current]
                               + self._total_cost(current, neighbor, obstacles))

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = (tentative_g
                                         + self._heuristic(neighbor, goal, obstacles))
                    parent[neighbor] = current
                    heapq.heappush(open_heap, (f_score[neighbor], neighbor))

        return []  # failure

    # ------------------------------------------------------------------
    # Cost components (Equations 14–21)
    # ------------------------------------------------------------------

    def _total_cost(self, ni, nj, obstacles) -> float:
        """Equation 14: C_total = C_base + C_penalty + C_semantic + C_smooth"""
        return (self._base_cost(ni, nj)
                + self._penalty_cost(nj)
                + self._semantic_cost(ni, nj)
                + self._smooth_cost(ni, nj))

    def _base_cost(self, ni, nj) -> float:
        """Equation 15: Euclidean distance for grid environments."""
        xi, yi = self._pos(ni)
        xj, yj = self._pos(nj)
        return math.hypot(xj - xi, yj - yi)

    def _penalty_cost(self, nj) -> float:
        """Equation 17: dynamic penalty from pre-computed penalty_map."""
        return self.penalty_map.get(nj, 0.0)

    def _semantic_cost(self, ni, nj) -> float:
        """Equation 19: semantic cost = W_base - sum(alpha_k * s_k)."""
        edge = (ni, nj)
        features = self.semantic_features.get(edge, np.zeros(len(self.alpha)))
        return self.w_base - float(np.dot(self.alpha, features))

    def _smooth_cost(self, ni, nj) -> float:
        """Equation 20: smoothness cost based on heading change."""
        if not self.current_path or len(self.current_path) < 2:
            return 0.0
        # Current heading
        prev = self.current_path[-2] if len(self.current_path) >= 2 else ni
        xi, yi = self._pos(prev)
        xc, yc = self._pos(ni)
        xj, yj = self._pos(nj)

        theta_current = math.atan2(yc - yi, xc - xi)
        theta_next    = math.atan2(yj - yc, xj - xc)
        delta = abs(theta_next - theta_current)
        # Wrap to [0, pi]
        delta = min(delta, 2 * math.pi - delta)
        return self.beta * delta

    # ------------------------------------------------------------------
    # Heuristic (Equation 22)
    # ------------------------------------------------------------------

    def _heuristic(self, node, goal, obstacles) -> float:
        """Equation 22: h(n,g) = h_euclidean * (1 + gamma * rho(n))"""
        xi, yi = self._pos(node)
        xg, yg = self._pos(goal)
        h_euc = math.hypot(xg - xi, yg - yi)
        rho   = self._obstacle_density(node, obstacles)
        inflation = min(self.gamma * rho, 0.5)   # hard cap (Section 3.2.2)
        return h_euc * (1.0 + inflation)

    def _obstacle_density(self, node, obstacles) -> float:
        """Equation 24: obstacles per unit area within r_sense."""
        xi, yi = self._pos(node)
        count = sum(
            1 for o in obstacles
            if math.hypot(o['px'] - xi, o['py'] - yi) <= self.r_sense
        )
        area = math.pi * self.r_sense ** 2
        return count / area if area > 0 else 0.0

    # ------------------------------------------------------------------
    # Replanning helpers (Equations 26–28)
    # ------------------------------------------------------------------

    def _select_replan_mode(self, current_pos, obstacles,
                            current_cost, initial_cost) -> str:
        """Equation 28: decision tree for replanning strategy."""
        # Minimum distance from current path to any obstacle
        min_dist = float('inf')
        for o in obstacles:
            for node in self.current_path:
                xi, yi = self._pos(node)
                d = math.hypot(o['px'] - xi, o['py'] - yi)
                min_dist = min(min_dist, d)

        cost_increase = ((current_cost / initial_cost - 1.0)
                         if initial_cost > 0 else 0.0)

        if min_dist < 3.0:
            return 'local'
        elif cost_increase > 0.25:
            return 'global'
        else:
            return 'smooth'

    def _smooth_path(self, path: List) -> List:
        """Post-process path: remove redundant waypoints (simple shortcutting)."""
        if len(path) < 3:
            return path
        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self._line_of_sight(path[i], path[j]):
                    break
                j -= 1
            smoothed.append(path[j])
            i = j
        return smoothed

    def _line_of_sight(self, a, b) -> bool:
        """Simple grid line-of-sight check (Bresenham)."""
        x0, y0 = self._pos(a)
        x1, y1 = self._pos(b)
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        err = dx - dy
        x, y = int(x0), int(y0)
        while True:
            node = (x, y)
            if self.penalty_map.get(node, 0) > self.lambda_base:
                return False
            if x == int(x1) and y == int(y1):
                return True
            e2 = 2 * err
            if e2 > -dy:
                err -= dy; x += sx
            if e2 < dx:
                err += dx; y += sy

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _neighbors(self, node) -> List:
        if hasattr(self.graph, 'neighbors'):
            return list(self.graph.neighbors(node))
        return self.graph.get(node, [])

    def _pos(self, node) -> Tuple[float, float]:
        """Extract (x, y) from node.  Supports tuple nodes or graph attr."""
        if isinstance(node, tuple):
            return float(node[0]), float(node[1])
        try:
            d = self.graph.nodes[node]
            return float(d.get('x', 0)), float(d.get('y', 0))
        except Exception:
            return 0.0, 0.0

    def _reconstruct_path(self, parent: Dict, current) -> List:
        path = []
        while current is not None:
            path.append(current)
            current = parent[current]
        return list(reversed(path))

    def _best_partial_path(self, parent: Dict, g_score: Dict, goal) -> List:
        """Return path to node closest to goal when time budget exceeded."""
        best = min(g_score, key=lambda n: self._base_cost(n, goal))
        return self._reconstruct_path(parent, best)

    def _path_cost(self, path: List, obstacles: List[Dict]) -> float:
        if len(path) < 2:
            return 0.0
        return sum(self._total_cost(path[i], path[i+1], obstacles)
                   for i in range(len(path) - 1))

    def set_semantic_features(self, features: Dict):
        """Set edge semantic features dict: {(ni, nj): np.ndarray}"""
        self.semantic_features = features
