"""
Baseline path planning algorithms for comparative evaluation.
Section 4.3: Standard A*, Dijkstra, RRT*, ACO.
"""

import heapq
import math
import random
import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Set


# ---------------------------------------------------------------------------
# Standard A*
# ---------------------------------------------------------------------------

class StandardAStar:
    """Standard A* with Euclidean heuristic (baseline [2])."""

    def __init__(self, graph, t_max: float = 5.0):
        self.graph = graph
        self.t_max = t_max

    def plan(self, start, goal) -> Tuple[List, int]:
        """Returns (path, nodes_explored)."""
        t0 = time.perf_counter()
        open_heap: List = []
        g: Dict  = {start: 0.0}
        parent: Dict = {start: None}
        closed: Set  = set()
        nodes_explored = 0

        heapq.heappush(open_heap, (self._h(start, goal), start))

        while open_heap:
            if time.perf_counter() - t0 > self.t_max:
                break
            _, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            closed.add(cur)
            nodes_explored += 1

            if cur == goal:
                return self._path(parent, cur), nodes_explored

            for nb in self._neighbors(cur):
                tg = g[cur] + self._dist(cur, nb)
                if nb not in g or tg < g[nb]:
                    g[nb] = tg
                    parent[nb] = cur
                    heapq.heappush(open_heap, (tg + self._h(nb, goal), nb))

        return [], nodes_explored

    def _h(self, a, b) -> float:
        xa, ya = self._pos(a)
        xb, yb = self._pos(b)
        return math.hypot(xb - xa, yb - ya)

    def _dist(self, a, b) -> float:
        return self._h(a, b)

    def _neighbors(self, node) -> List:
        if hasattr(self.graph, 'neighbors'):
            return list(self.graph.neighbors(node))
        return self.graph.get(node, [])

    def _pos(self, node) -> Tuple[float, float]:
        if isinstance(node, tuple):
            return float(node[0]), float(node[1])
        try:
            d = self.graph.nodes[node]
            return float(d.get('x', 0)), float(d.get('y', 0))
        except Exception:
            return 0.0, 0.0

    def _path(self, parent: Dict, cur) -> List:
        path = []
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        return list(reversed(path))


# ---------------------------------------------------------------------------
# Dijkstra
# ---------------------------------------------------------------------------

class Dijkstra:
    """Dijkstra's algorithm (baseline [1])."""

    def __init__(self, graph):
        self.graph = graph

    def plan(self, start, goal) -> Tuple[List, int]:
        dist   = {start: 0.0}
        parent = {start: None}
        heap   = [(0.0, start)]
        visited: Set = set()
        nodes_explored = 0

        while heap:
            d, cur = heapq.heappop(heap)
            if cur in visited:
                continue
            visited.add(cur)
            nodes_explored += 1

            if cur == goal:
                return self._path(parent, cur), nodes_explored

            for nb in self._neighbors(cur):
                nd = d + self._dist(cur, nb)
                if nb not in dist or nd < dist[nb]:
                    dist[nb] = nd
                    parent[nb] = cur
                    heapq.heappush(heap, (nd, nb))

        return [], nodes_explored

    def _dist(self, a, b) -> float:
        if isinstance(a, tuple) and isinstance(b, tuple):
            return math.hypot(b[0]-a[0], b[1]-a[1])
        try:
            d = self.graph.nodes
            xa, ya = float(d[a].get('x',0)), float(d[a].get('y',0))
            xb, yb = float(d[b].get('x',0)), float(d[b].get('y',0))
            return math.hypot(xb-xa, yb-ya)
        except Exception:
            return 1.0

    def _neighbors(self, node) -> List:
        if hasattr(self.graph, 'neighbors'):
            return list(self.graph.neighbors(node))
        return self.graph.get(node, [])

    def _path(self, parent: Dict, cur) -> List:
        path = []
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        return list(reversed(path))


# ---------------------------------------------------------------------------
# RRT*
# ---------------------------------------------------------------------------

class RRTStar:
    """
    RRT* (Rapidly-exploring Random Tree Star) [4].
    Section 4.3: 10,000 iterations, goal bias 0.05.
    """

    def __init__(self, bounds: Tuple[float, float, float, float],
                 graph_nodes: List,
                 max_iter: int = 10000,
                 goal_bias: float = 0.05,
                 step_size: float = 2.0,
                 gamma_rrt: float = 2.5,
                 dim: int = 2):
        """bounds = (x_min, y_min, x_max, y_max)"""
        self.bounds    = bounds
        self.nodes_ref = graph_nodes   # for feasibility checking
        self.max_iter  = max_iter
        self.goal_bias = goal_bias
        self.step      = step_size
        self.gamma     = gamma_rrt
        self.dim       = dim

    def plan(self, start: Tuple, goal: Tuple,
             obstacle_checker=None) -> Tuple[List, int]:
        """
        Returns (path as list of (x,y) tuples, iterations_used).
        obstacle_checker: callable(x, y) -> bool (True = free)
        """
        nodes   = [start]
        parent  = {start: None}
        costs   = {start: 0.0}

        for i in range(self.max_iter):
            # Sample
            if random.random() < self.goal_bias:
                sample = goal
            else:
                sample = self._random_sample()

            nearest = self._nearest(nodes, sample)
            new_node = self._steer(nearest, sample)

            if obstacle_checker and not obstacle_checker(*new_node):
                continue

            # Rewiring radius
            n   = len(nodes)
            r   = self.gamma * math.sqrt(math.log(n + 1) / (n + 1))

            near = self._near(nodes, new_node, r)
            best_parent, best_cost = nearest, costs[nearest] + self._dist(nearest, new_node)

            for nb in near:
                c = costs[nb] + self._dist(nb, new_node)
                if c < best_cost:
                    if obstacle_checker is None or obstacle_checker(*new_node):
                        best_parent, best_cost = nb, c

            nodes.append(new_node)
            parent[new_node] = best_parent
            costs[new_node]  = best_cost

            # Rewire
            for nb in near:
                c = best_cost + self._dist(new_node, nb)
                if c < costs.get(nb, float('inf')):
                    if obstacle_checker is None or obstacle_checker(*nb):
                        parent[nb] = new_node
                        costs[nb]  = c

            # Check goal reach
            if self._dist(new_node, goal) < self.step:
                parent[goal] = new_node
                costs[goal]  = best_cost + self._dist(new_node, goal)
                nodes.append(goal)
                return self._extract_path(parent, goal), i + 1

        # Return path to closest node
        closest = min(nodes, key=lambda n: self._dist(n, goal))
        return self._extract_path(parent, closest), self.max_iter

    def _random_sample(self) -> Tuple[float, float]:
        xmin, ymin, xmax, ymax = self.bounds
        return (random.uniform(xmin, xmax), random.uniform(ymin, ymax))

    def _nearest(self, nodes, sample) -> Tuple:
        return min(nodes, key=lambda n: self._dist(n, sample))

    def _near(self, nodes, node, r) -> List:
        return [n for n in nodes if self._dist(n, node) <= r]

    def _steer(self, frm, to) -> Tuple[float, float]:
        d = self._dist(frm, to)
        if d < self.step:
            return to
        t = self.step / d
        return (frm[0] + t*(to[0]-frm[0]), frm[1] + t*(to[1]-frm[1]))

    def _dist(self, a, b) -> float:
        return math.hypot(b[0]-a[0], b[1]-a[1])

    def _extract_path(self, parent, goal) -> List:
        path = []
        cur  = goal
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)
        return list(reversed(path))


# ---------------------------------------------------------------------------
# Ant Colony Optimization (ACO)
# ---------------------------------------------------------------------------

class ACO:
    """
    Ant Colony Optimization [14].
    Section 4.3: 50 ants, 100 iterations, rho=0.1, alpha=1, beta=2.
    """

    def __init__(self, graph,
                 n_ants: int = 50,
                 n_iter: int = 100,
                 rho:    float = 0.1,
                 alpha:  float = 1.0,
                 beta:   float = 2.0,
                 q:      float = 1.0):
        self.graph  = graph
        self.n_ants = n_ants
        self.n_iter = n_iter
        self.rho    = rho
        self.alpha  = alpha
        self.beta   = beta
        self.q      = q

        # Initialise pheromones
        self.pheromone: Dict = {}

    def plan(self, start, goal) -> Tuple[List, int]:
        nodes = (list(self.graph.nodes())
                 if hasattr(self.graph, 'nodes')
                 else list(self.graph.keys()))

        # Initialise pheromone on all edges
        for node in nodes:
            for nb in self._neighbors(node):
                self.pheromone[(node, nb)] = 1.0

        best_path, best_cost = [], float('inf')

        for iteration in range(self.n_iter):
            all_paths  = []
            all_costs  = []

            for ant in range(self.n_ants):
                path = self._construct_path(start, goal)
                if path and path[-1] == goal:
                    cost = self._path_cost(path)
                    all_paths.append(path)
                    all_costs.append(cost)
                    if cost < best_cost:
                        best_cost, best_path = cost, path

            # Evaporate
            for key in self.pheromone:
                self.pheromone[key] *= (1 - self.rho)

            # Deposit
            for path, cost in zip(all_paths, all_costs):
                deposit = self.q / cost if cost > 0 else 0
                for i in range(len(path) - 1):
                    edge = (path[i], path[i+1])
                    self.pheromone[edge] = self.pheromone.get(edge, 0) + deposit

        return best_path, self.n_ants * self.n_iter

    def _construct_path(self, start, goal) -> List:
        path    = [start]
        visited = {start}
        cur     = start

        for _ in range(500):   # max steps
            if cur == goal:
                break
            nbrs = [n for n in self._neighbors(cur) if n not in visited]
            if not nbrs:
                break

            probs = self._transition_probs(cur, nbrs, goal)
            cur   = random.choices(nbrs, weights=probs, k=1)[0]
            path.append(cur)
            visited.add(cur)

        return path

    def _transition_probs(self, cur, neighbors, goal) -> List[float]:
        scores = []
        for nb in neighbors:
            tau  = self.pheromone.get((cur, nb), 1.0) ** self.alpha
            eta  = (1.0 / max(self._dist(nb, goal), 1e-9)) ** self.beta
            scores.append(tau * eta)
        total = sum(scores)
        return [s / total for s in scores] if total > 0 else [1/len(scores)] * len(scores)

    def _path_cost(self, path: List) -> float:
        return sum(self._dist(path[i], path[i+1]) for i in range(len(path)-1))

    def _dist(self, a, b) -> float:
        if isinstance(a, tuple) and isinstance(b, tuple):
            return math.hypot(b[0]-a[0], b[1]-a[1])
        try:
            d = self.graph.nodes
            xa, ya = float(d[a].get('x',0)), float(d[a].get('y',0))
            xb, yb = float(d[b].get('x',0)), float(d[b].get('y',0))
            return math.hypot(xb-xa, yb-ya)
        except Exception:
            return 1.0

    def _neighbors(self, node) -> List:
        if hasattr(self.graph, 'neighbors'):
            return list(self.graph.neighbors(node))
        return self.graph.get(node, [])
