"""
Environment representation and mapping (Module 1).
Supports synthetic grid environments and real-world OSM road networks.
"""

import math
import random
import numpy as np
from typing import List, Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# Cell states
# ---------------------------------------------------------------------------
FREE     = 0
OBSTACLE = 1
DYNAMIC  = 2


# ---------------------------------------------------------------------------
# Synthetic grid environment
# ---------------------------------------------------------------------------

class GridEnvironment:
    """
    2-D grid environment with static and dynamic obstacles.
    Cell resolution r = 0.5–1.0 m (default 1.0 m).
    """

    def __init__(self, width: int, height: int,
                 obstacle_density: float = 0.2,
                 n_dynamic: int = 10,
                 resolution: float = 1.0,
                 seed: Optional[int] = None):
        """
        Parameters
        ----------
        width, height      : grid dimensions in cells
        obstacle_density   : fraction of cells as static obstacles (0–1)
        n_dynamic          : number of dynamic obstacles
        resolution         : metres per cell
        seed               : random seed for reproducibility
        """
        self.W   = width
        self.H   = height
        self.rho = obstacle_density
        self.res = resolution
        self.rng = random.Random(seed)

        # Grid: 0=FREE, 1=OBSTACLE, 2=DYNAMIC
        self.grid = np.zeros((height, width), dtype=np.uint8)
        self._place_static_obstacles()

        # Dynamic obstacles: list of dicts with state
        self.dynamic_obstacles: List[Dict] = []
        self._init_dynamic_obstacles(n_dynamic)

        # Graph representation (adjacency for A*)
        self.graph = self._build_adjacency()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _place_static_obstacles(self):
        n_obs = int(self.W * self.H * self.rho)
        positions = self.rng.sample(
            [(x, y) for x in range(self.W) for y in range(self.H)],
            n_obs
        )
        for x, y in positions:
            self.grid[y][x] = OBSTACLE

    def _init_dynamic_obstacles(self, n: int):
        """
        Section 4.2.1: velocities sampled from N(2.5, 0.5^2) m/s,
        random headings, constant velocity + Gaussian process noise.
        """
        for i in range(n):
            while True:
                x = self.rng.randint(0, self.W - 1)
                y = self.rng.randint(0, self.H - 1)
                if self.grid[y][x] == FREE:
                    break
            speed   = max(0.1, self.rng.gauss(2.5, 0.5))
            heading = self.rng.uniform(0, 2 * math.pi)
            self.dynamic_obstacles.append({
                'id':  f'dyn_{i}',
                'px':  float(x),
                'py':  float(y),
                'vx':  speed * math.cos(heading),
                'vy':  speed * math.sin(heading),
            })

    def _build_adjacency(self) -> Dict:
        """8-connected adjacency, excluding static obstacles."""
        adj: Dict = {}
        for y in range(self.H):
            for x in range(self.W):
                if self.grid[y][x] == OBSTACLE:
                    continue
                node = (x, y)
                neighbors = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.W and 0 <= ny < self.H:
                            if self.grid[ny][nx] != OBSTACLE:
                                neighbors.append((nx, ny))
                adj[node] = neighbors
        return adj

    # ------------------------------------------------------------------
    # Random start / goal (Section 4.2.1: min 70% diagonal separation)
    # ------------------------------------------------------------------

    def sample_start_goal(self) -> Tuple[Tuple, Tuple]:
        diag = math.hypot(self.W, self.H)
        min_sep = 0.7 * diag
        free_cells = [(x, y)
                      for x in range(self.W) for y in range(self.H)
                      if self.grid[y][x] == FREE]
        while True:
            start = self.rng.choice(free_cells)
            goal  = self.rng.choice(free_cells)
            if math.hypot(goal[0]-start[0], goal[1]-start[1]) >= min_sep:
                return start, goal

    # ------------------------------------------------------------------
    # Step simulation (constant velocity + boundary reflection)
    # ------------------------------------------------------------------

    def step(self, dt: float = 0.1, noise_std: float = 0.3):
        """
        Advance dynamic obstacles by dt seconds.
        Obstacles bounce off walls and avoid static obstacles.
        """
        for obs in self.dynamic_obstacles:
            # Add process noise
            obs['vx'] += self.rng.gauss(0, noise_std) * dt
            obs['vy'] += self.rng.gauss(0, noise_std) * dt

            nx = obs['px'] + obs['vx'] * dt
            ny = obs['py'] + obs['vy'] * dt

            # Boundary reflection
            if nx < 0 or nx >= self.W:
                obs['vx'] *= -1
                nx = obs['px']
            if ny < 0 or ny >= self.H:
                obs['vy'] *= -1
                ny = obs['py']

            # Avoid static obstacles
            gx, gy = int(round(nx)), int(round(ny))
            if (0 <= gx < self.W and 0 <= gy < self.H
                    and self.grid[gy][gx] == OBSTACLE):
                obs['vx'] *= -1
                obs['vy'] *= -1
            else:
                obs['px'] = nx
                obs['py'] = ny

    def get_obstacle_measurements(self) -> List[Dict]:
        """Return current obstacle positions as measurement dicts."""
        return [{'id': o['id'],
                 'px': o['px'],
                 'py': o['py'],
                 'measurement': np.array([o['px'], o['py']])}
                for o in self.dynamic_obstacles]

    def node_position(self, node: Tuple) -> Tuple[float, float]:
        return float(node[0]), float(node[1])

    @property
    def all_nodes(self) -> List:
        return list(self.graph.keys())


# ---------------------------------------------------------------------------
# OSM road network environment
# ---------------------------------------------------------------------------

class OSMEnvironment:
    """
    Real-world road network from OpenStreetMap via OSMnx.
    Section 4.2.2.
    """

    def __init__(self, location: str,
                 dist: int = 1000,
                 network_type: str = 'drive'):
        """
        Parameters
        ----------
        location     : place name string, e.g. 'Kattankulathur, Chennai, India'
        dist         : radius in metres around the centre point
        network_type : 'drive' | 'walk' | 'bike' | 'all'
        """
        try:
            import osmnx as ox
        except ImportError:
            raise ImportError("osmnx is required for OSM environments: pip install osmnx")

        self.graph = ox.graph_from_place(location, network_type=network_type)
        self.graph = ox.project_graph(self.graph)   # project to UTM
        self.nodes_list = list(self.graph.nodes())

    def sample_start_goal(self) -> Tuple[int, int]:
        """Sample random connected start/goal pair."""
        import random
        nodes = self.nodes_list
        while True:
            start = random.choice(nodes)
            goal  = random.choice(nodes)
            if start != goal:
                return start, goal

    def node_position(self, node: int) -> Tuple[float, float]:
        d = self.graph.nodes[node]
        return float(d.get('x', 0)), float(d.get('y', 0))

    @property
    def all_nodes(self) -> List:
        return self.nodes_list

    def simulate_traffic(self, n_vehicles: int = 15,
                         speed_noise_std: float = 0.2) -> List[Dict]:
        """
        Simulate dynamic obstacles using traffic-like speeds.
        Section 4.2.2: speeds ~ posted limits + Gaussian noise.
        """
        import random
        obstacles = []
        edges = list(self.graph.edges(data=True))
        for i in range(n_vehicles):
            if not edges:
                break
            u, v, data = random.choice(edges)
            xu, yu = self.node_position(u)
            speed_limit = float(str(data.get('maxspeed', '50'))
                                .replace(' km/h','').split(';')[0])
            speed = max(1.0, speed_limit + random.gauss(0, speed_noise_std * speed_limit))
            # Convert km/h to m/s
            speed_ms = speed / 3.6
            xv, yv = self.node_position(v)
            dx = xv - xu
            dy = yv - yu
            length = math.hypot(dx, dy) or 1.0
            obstacles.append({
                'id':  f'vehicle_{i}',
                'px':  xu,
                'py':  yu,
                'vx':  speed_ms * dx / length,
                'vy':  speed_ms * dy / length,
                'measurement': np.array([xu, yu]),
            })
        return obstacles
