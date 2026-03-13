"""
Dynamic obstacle detection and prediction module.
Implements Kalman filter trajectory prediction (Equations 2–8, Algorithm 2).
"""

import time
import math
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple


class KalmanTracker:
    """
    Constant-velocity Kalman filter for a single obstacle.

    State vector: x = [px, py, vx, vy]^T   (Equation 2)
    """

    # Default noise covariances (Section 3.1, Module 2)
    Q_DIAG = [0.1, 0.1, 0.5, 0.5]   # Process noise
    R_DIAG = [0.5, 0.5]              # Measurement noise

    def __init__(self, initial_state: np.ndarray, dt: float = 0.1):
        """
        Parameters
        ----------
        initial_state : [px, py, vx, vy]
        dt            : time step (seconds)
        """
        self.dt = dt
        n = 4   # state dimension
        m = 2   # measurement dimension (px, py)

        # State transition matrix F  (Equation 3)
        self.F = np.array([
            [1, 0, dt, 0 ],
            [0, 1, 0,  dt],
            [0, 0, 1,  0 ],
            [0, 0, 0,  1 ],
        ], dtype=float)

        # Observation matrix H
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        # Noise covariances
        self.Q = np.diag(self.Q_DIAG)   # process noise
        self.R = np.diag(self.R_DIAG)   # measurement noise

        # Initial state and covariance
        self.x = initial_state.astype(float).reshape(n, 1)
        self.P = np.eye(n)

    # ------------------------------------------------------------------
    # Kalman equations (Equations 4–8)
    # ------------------------------------------------------------------

    def predict(self):
        """Prediction step (Equations 4–5)."""
        self.x = self.F @ self.x                      # Eq. 4
        self.P = self.F @ self.P @ self.F.T + self.Q  # Eq. 5

    def update(self, measurement: np.ndarray):
        """Update step (Equations 6–8)."""
        z = measurement.reshape(2, 1)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)            # Eq. 6
        self.x = self.x + K @ (z - self.H @ self.x)         # Eq. 7
        self.P = (np.eye(4) - K @ self.H) @ self.P          # Eq. 8

    def get_state(self) -> np.ndarray:
        """Return current state [px, py, vx, vy]."""
        return self.x.flatten()

    def predict_trajectory(self, horizon: float) -> List[np.ndarray]:
        """
        Predict future positions over horizon seconds.

        Returns list of [px, py] arrays at each dt step.
        """
        steps = int(horizon / self.dt)
        x_sim = self.x.copy()
        trajectory = []
        for _ in range(steps):
            x_sim = self.F @ x_sim
            trajectory.append(x_sim[:2].flatten())
        return trajectory


# ---------------------------------------------------------------------------

class DynamicPenaltyMap:
    """
    Asynchronous penalty map updater (Algorithm 2).
    Runs at f = 10 Hz in a background thread.
    """

    def __init__(self, graph_nodes: List,
                 update_freq: float = 10.0,
                 decay_rate:  float = 0.05,
                 pred_horizon: float = 2.0,
                 lambda_base: float = 5.0,
                 sigma:       float = 2.0,
                 v_max:       float = 10.0,
                 active_radius_multiplier: float = 3.0):

        self.nodes    = graph_nodes
        self.freq     = update_freq
        self.delta    = 1.0 / update_freq        # 0.1 s
        self.decay    = decay_rate               # δ  (τ = 1/δ = 20 s)
        self.horizon  = pred_horizon
        self.lb       = lambda_base
        self.sigma    = sigma
        self.v_max    = v_max
        self.arm      = active_radius_multiplier

        # Shared state
        self.penalty_map: Dict = {n: 0.0 for n in graph_nodes}
        self.trackers:    Dict = {}              # obstacle_id -> KalmanTracker
        self._lock        = threading.Lock()
        self._running     = False
        self._thread: Optional[threading.Thread] = None

        # Active region (subset of nodes near current path)
        self._active_nodes: List = list(graph_nodes)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self):
        """Start background update thread."""
        self._running = True
        self._thread  = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def update_obstacle(self, obs_id: str, measurement: np.ndarray):
        """
        Feed a new position measurement [px, py] for an obstacle.
        Creates tracker on first call.
        """
        with self._lock:
            if obs_id not in self.trackers:
                state = np.array([measurement[0], measurement[1], 0.0, 0.0])
                self.trackers[obs_id] = KalmanTracker(state, dt=self.delta)
            else:
                self.trackers[obs_id].predict()
                self.trackers[obs_id].update(measurement)

    def set_active_region(self, path_nodes: List, path_length: float,
                          start, goal, node_pos_fn):
        """
        Restrict penalty updates to nodes within 3 * path_length of current path.
        Equation 25.
        """
        if path_length == 0:
            # Initialisation: use straight-line distance start->goal
            sx, sy = node_pos_fn(start)
            gx, gy = node_pos_fn(goal)
            d_init  = math.hypot(gx - sx, gy - sy)
            radius  = self.arm * d_init
        else:
            radius  = self.arm * path_length

        path_set = set(path_nodes)
        with self._lock:
            self._active_nodes = [
                n for n in self.nodes
                if any(
                    _dist(node_pos_fn(n), node_pos_fn(p)) <= radius
                    for p in path_set
                )
            ]

    def get_map(self) -> Dict:
        with self._lock:
            return dict(self.penalty_map)

    # ------------------------------------------------------------------
    # Background loop (Algorithm 2)
    # ------------------------------------------------------------------

    def _update_loop(self):
        while self._running:
            t0 = time.perf_counter()
            self._compute_penalties()
            elapsed = time.perf_counter() - t0
            sleep_t = max(0.0, self.delta - elapsed)
            time.sleep(sleep_t)

    def _compute_penalties(self):
        with self._lock:
            active = list(self._active_nodes)
            trackers = dict(self.trackers)

        new_map: Dict = {n: 0.0 for n in active}

        for obs_id, tracker in trackers.items():
            # Predict trajectory (Algorithm 2, line 19)
            trajectory = tracker.predict_trajectory(self.horizon)
            state = tracker.get_state()
            speed = math.hypot(state[2], state[3])

            # Obstacle weight: higher for faster obstacles (Equation 18)
            lam = self.lb * (1.0 + speed / self.v_max)

            for n in active:
                px_node, py_node = _node_xy(n)

                # Minimum distance to predicted trajectory
                d_min = min(
                    math.hypot(tp[0] - px_node, tp[1] - py_node)
                    for tp in trajectory
                ) if trajectory else float('inf')

                # Penalty contribution (Equation 17)
                penalty = lam * math.exp(-d_min ** 2 / (2 * self.sigma ** 2))
                new_map[n] = new_map.get(n, 0.0) + penalty

        # Temporal decay on existing penalties (Algorithm 2, lines 33–34)
        with self._lock:
            for n in active:
                old = self.penalty_map.get(n, 0.0)
                decayed = old * math.exp(-self.decay * self.delta)
                self.penalty_map[n] = decayed + new_map.get(n, 0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node_xy(node) -> Tuple[float, float]:
    if isinstance(node, tuple):
        return float(node[0]), float(node[1])
    return 0.0, 0.0


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])
