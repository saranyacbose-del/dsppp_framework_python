"""
Experiment harness — reproduces paper results (Section 4–5).
Runs DSPPP vs baselines across grid sizes, densities, and OSM networks.
"""

import time
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from environment.grid_env import GridEnvironment
from algorithms.dsppp import DSPPPPlanner
from algorithms.kalman_obstacle import DynamicPenaltyMap
from algorithms.baselines import StandardAStar, Dijkstra, RRTStar, ACO
from semantic.irl_preference import IRLPreferenceLearner, extract_semantic_features
from utils.metrics import (compute_ppe, path_length, path_smoothness,
                            safety_margin, cohens_d, bootstrap_ci,
                            one_way_anova, effect_size_label)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GRID_CONFIGS = [
    {'width': 50,  'height': 50,  'label': '50x50'},
    {'width': 100, 'height': 100, 'label': '100x100'},
    {'width': 200, 'height': 200, 'label': '200x200'},
]

DENSITIES    = [0.10, 0.20, 0.30, 0.40]
N_DYNAMIC    = [5, 10, 15, 20]       # mapped to densities
N_TRIALS     = 30
PROFILES     = ['speed', 'safety', 'balanced']   # 10 trials each


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_single_trial(width: int, height: int,
                     density: float, n_dyn: int,
                     profile: str, seed: int,
                     params: Optional[Dict] = None) -> Dict:
    """
    Run one planning trial for all algorithms.
    Returns dict of metrics per algorithm.
    """
    env = GridEnvironment(width, height, density, n_dyn,
                          resolution=1.0, seed=seed)
    start, goal = env.sample_start_goal()

    # IRL alpha
    learner = IRLPreferenceLearner(profile=profile)
    alpha   = learner.alpha

    # Dynamic penalty map
    dpm = DynamicPenaltyMap(
        graph_nodes=env.all_nodes,
        **(params or {})
    )
    # Pre-fill penalty map with current obstacle positions
    for obs in env.dynamic_obstacles:
        meas = np.array([obs['px'], obs['py']])
        dpm.update_obstacle(obs['id'], meas)
    dpm._compute_penalties()
    penalty_map = dpm.get_map()

    obstacles = env.get_obstacle_measurements()

    results = {}

    # ---- DSPPP ----
    planner = DSPPPPlanner(env.graph, penalty_map, alpha, params)
    t0 = time.perf_counter()
    path, cost = planner.plan(start, goal, obstacles)
    t_ms = (time.perf_counter() - t0) * 1000

    results['DSPPP'] = _collect_metrics(
        path, cost, t_ms, env, alpha, obstacles
    )

    # ---- Standard A* ----
    astar = StandardAStar(env.graph)
    t0    = time.perf_counter()
    path_a, nodes_exp = astar.plan(start, goal)
    t_ms  = (time.perf_counter() - t0) * 1000
    results['A*'] = _collect_metrics(
        path_a, None, t_ms, env, alpha, obstacles,
        nodes_explored=nodes_exp
    )

    # ---- Dijkstra ----
    dijk = Dijkstra(env.graph)
    t0   = time.perf_counter()
    path_d, nodes_exp = dijk.plan(start, goal)
    t_ms = (time.perf_counter() - t0) * 1000
    results['Dijkstra'] = _collect_metrics(
        path_d, None, t_ms, env, alpha, obstacles,
        nodes_explored=nodes_exp
    )

    # ---- RRT* ----
    bounds = (0, 0, width - 1, height - 1)
    rrt    = RRTStar(bounds, env.all_nodes)
    free_check = lambda x, y: (
        0 <= int(x) < width and 0 <= int(y) < height
        and env.grid[int(y)][int(x)] == 0
    )
    t0 = time.perf_counter()
    path_r, iters = rrt.plan(
        (float(start[0]), float(start[1])),
        (float(goal[0]),  float(goal[1])),
        obstacle_checker=free_check
    )
    t_ms = (time.perf_counter() - t0) * 1000
    # Convert float tuples back to grid tuples
    path_r_grid = [(int(round(p[0])), int(round(p[1]))) for p in path_r]
    results['RRT*'] = _collect_metrics(
        path_r_grid, None, t_ms, env, alpha, obstacles,
        nodes_explored=iters
    )

    # ---- ACO ----
    aco  = ACO(env.graph)
    t0   = time.perf_counter()
    path_aco, iters_aco = aco.plan(start, goal)
    t_ms = (time.perf_counter() - t0) * 1000
    results['ACO'] = _collect_metrics(
        path_aco, None, t_ms, env, alpha, obstacles,
        nodes_explored=iters_aco
    )

    return results


def _collect_metrics(path, cost, t_ms, env, alpha, obstacles,
                     nodes_explored: int = 0) -> Dict:
    """Compute all 7 evaluation metrics for a path."""
    success = len(path) >= 2
    feature_dict = {}   # empty for grid (no OSM semantics)

    return {
        'time_ms':        t_ms,
        'path_length':    path_length(path, env.node_position) if success else 0,
        'success':        int(success),
        'smoothness':     path_smoothness(path, env.node_position) if success else 180.0,
        'nodes_explored': nodes_explored,
        'safety':         safety_margin(path, env.node_position, obstacles) if success else 0,
        'ppe':            compute_ppe(path, env.node_position, alpha,
                                      feature_dict, obstacles, t_ms) if success else 0,
    }


# ---------------------------------------------------------------------------
# Full experiment (30 trials × configs)
# ---------------------------------------------------------------------------

def run_experiment(grid_label: str = '100x100',
                   density: float = 0.20,
                   n_trials: int = N_TRIALS,
                   output_path: Optional[str] = None) -> Dict:
    """
    Reproduce Table II results for a given grid config and density.
    """
    cfg = next((c for c in GRID_CONFIGS if c['label'] == grid_label), GRID_CONFIGS[1])
    w, h = cfg['width'], cfg['height']

    all_results: Dict[str, List[Dict]] = {
        'DSPPP': [], 'A*': [], 'Dijkstra': [], 'RRT*': [], 'ACO': []
    }

    profiles_cycle = PROFILES * (n_trials // len(PROFILES) + 1)

    print(f"\n{'='*60}")
    print(f"Grid: {grid_label}  |  Density: {density:.0%}  |  Trials: {n_trials}")
    print(f"{'='*60}")

    for trial in range(n_trials):
        profile = profiles_cycle[trial]
        seed    = trial * 137 + 42
        n_dyn   = N_DYNAMIC[DENSITIES.index(density)] if density in DENSITIES else 10

        trial_results = run_single_trial(w, h, density, n_dyn,
                                          profile, seed)
        for algo, metrics in trial_results.items():
            all_results[algo].append(metrics)

        if (trial + 1) % 5 == 0:
            print(f"  Trial {trial+1}/{n_trials} done")

    # Aggregate
    summary = _aggregate(all_results)
    _print_table(summary, grid_label, density)
    _run_statistics(all_results)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({'config': {'grid': grid_label, 'density': density},
                       'raw': {k: v for k, v in all_results.items()},
                       'summary': summary}, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return summary


def _aggregate(all_results: Dict) -> Dict:
    summary = {}
    for algo, trials in all_results.items():
        if not trials:
            continue
        keys = trials[0].keys()
        summary[algo] = {}
        for k in keys:
            vals = np.array([t[k] for t in trials])
            summary[algo][k] = {
                'mean': float(np.mean(vals)),
                'std':  float(np.std(vals, ddof=1)),
            }
        summary[algo]['success_rate'] = float(
            np.mean([t['success'] for t in trials]) * 100
        )
    return summary


def _print_table(summary: Dict, label: str, density: float):
    print(f"\nResults: {label} grid, {density:.0%} obstacles")
    hdr = f"{'Algorithm':<15} {'Time(ms)':>10} {'Length':>8} {'Success%':>9} {'Smooth°':>8} {'Safety':>8} {'PPE':>10}"
    print(hdr)
    print('-' * len(hdr))
    for algo, s in summary.items():
        print(
            f"{algo:<15}"
            f" {s['time_ms']['mean']:>8.1f}±{s['time_ms']['std']:.1f}"
            f" {s['path_length']['mean']:>8.1f}"
            f" {s['success_rate']:>9.1f}"
            f" {s['smoothness']['mean']:>7.1f}°"
            f" {s['safety']['mean']:>8.2f}"
            f" {s['ppe']['mean']:>10.4f}"
        )


def _run_statistics(all_results: Dict):
    """ANOVA + Cohen's d for computation time (Table IV)."""
    try:
        from scipy import stats
        groups = {k: np.array([t['time_ms'] for t in v])
                  for k, v in all_results.items() if v}
        if len(groups) < 2:
            return

        F, p = stats.f_oneway(*groups.values())
        print(f"\nANOVA (time_ms): F={F:.1f}, p={p:.2e}")

        dsppp_t = groups.get('DSPPP', np.array([]))
        astar_t  = groups.get('A*',   np.array([]))
        if len(dsppp_t) > 1 and len(astar_t) > 1:
            d = cohens_d(dsppp_t, astar_t)
            ci_lo, ci_hi = bootstrap_ci(astar_t - dsppp_t)
            print(f"Cohen's d (DSPPP vs A*): {d:.2f} ({effect_size_label(d)})")
            print(f"95% CI for time difference: [{ci_lo:.1f}ms, {ci_hi:.1f}ms]")
    except ImportError:
        print("\n(scipy not available — skipping statistical tests)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DSPPP Experiment Runner')
    parser.add_argument('--grid',    default='100x100',
                        choices=[c['label'] for c in GRID_CONFIGS])
    parser.add_argument('--density', type=float, default=0.20)
    parser.add_argument('--trials',  type=int,   default=30)
    parser.add_argument('--output',  default='results/experiment_results.json')
    args = parser.parse_args()

    run_experiment(args.grid, args.density, args.trials, args.output)
