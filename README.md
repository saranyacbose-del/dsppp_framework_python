# DSPPP Framework
**Dynamic Semantic Personalized Path Planning**

Python implementation accompanying the paper:
> *"Semantic-Aware Personalized Path Planning for Autonomous Navigation with Real-Time Dynamic Obstacle Prediction"*
> Saranya C & Janaki G, SRM Institute of Science & Technology

---

## Package Structure

```
dsppp_framework/
├── algorithms/
│   ├── dsppp.py            # Improved A* (Algorithms 1 & 2, Equations 14–24)
│   ├── kalman_obstacle.py  # Kalman tracker + async penalty map (Equations 2–8)
│   └── baselines.py        # Standard A*, Dijkstra, RRT*, ACO
├── environment/
│   └── grid_env.py         # Synthetic grid + OSM environments (Section 4.2)
├── semantic/
│   └── irl_preference.py   # IRL preference learning (Equations 9–13)
├── utils/
│   └── metrics.py          # PPE metric + statistical analysis (Equations 29–32)
├── experiments/
│   └── run_experiment.py   # Full evaluation harness (Section 5)
├── demo.py                 # Quick sanity-check demo
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick Start

```bash
python demo.py
```

Expected output (50×50 grid, 20% obstacles):

```
Metric             DSPPP    Standard A*
------------------------------------------
Time ms            ~87        ~154
Length             ~152       ~156
Smoothness°        ~12.4      ~18.7
Safety             ~2.3       ~1.9
PPE                ~0.0423    ~0.0287
```

---

## Run Full Experiments

```bash
# Reproduce Table II (100×100 grid, 20% obstacles, 30 trials)
python experiments/run_experiment.py --grid 100x100 --density 0.20 --trials 30 --output results/table2.json

# Scalability test (200×200)
python experiments/run_experiment.py --grid 200x200 --density 0.20 --trials 10
```

---

## Key Classes

### `DSPPPPlanner`  (`algorithms/dsppp.py`)
```python
from algorithms.dsppp import DSPPPPlanner

planner = DSPPPPlanner(graph, penalty_map, alpha)
path, cost = planner.plan(start, goal, obstacles)
```

### `DynamicPenaltyMap`  (`algorithms/kalman_obstacle.py`)
```python
from algorithms.kalman_obstacle import DynamicPenaltyMap

dpm = DynamicPenaltyMap(graph_nodes=nodes, update_freq=10.0)
dpm.start()                                     # 10 Hz background thread
dpm.update_obstacle('obs_0', np.array([x, y]))  # feed measurements
penalty_map = dpm.get_map()
dpm.stop()
```

### `IRLPreferenceLearner`  (`semantic/irl_preference.py`)
```python
from semantic.irl_preference import IRLPreferenceLearner

learner = IRLPreferenceLearner(profile='speed')   # or 'safety', 'balanced'
alpha = learner.fit(observed_routes, feature_dict, graph=G)
```

### PPE Metric  (`utils/metrics.py`)
```python
from utils.metrics import compute_ppe

ppe = compute_ppe(path, node_pos_fn, alpha, feature_dict,
                  obstacles, computation_time_ms=87.3)
```

---

## ROS2 Integration

The planner is designed for drop-in use with `nav2_core::GlobalPlanner`.
Set the penalty map update thread to run alongside the ROS2 spin loop:

```python
dpm.start()   # async 10 Hz thread
# ... ROS2 node spin ...
dpm.stop()
```

---

## Citation

```bibtex
@article{saranya2025dsppp,
  title   = {Semantic-Aware Personalized Path Planning for Autonomous Navigation
             with Real-Time Dynamic Obstacle Prediction},
  author  = {Saranya, C and Janaki, G},
  journal = {SRM Institute of Science and Technology},
  year    = {2025}
}
```

---

## License

MIT License — see LICENSE file.
