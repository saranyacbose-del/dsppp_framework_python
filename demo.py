"""
Quick demo: run DSPPP on a 50x50 grid and print results.
Usage: python demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np

from environment.grid_env import GridEnvironment
from algorithms.dsppp import DSPPPPlanner
from algorithms.kalman_obstacle import DynamicPenaltyMap
from algorithms.baselines import StandardAStar
from semantic.irl_preference import IRLPreferenceLearner
from utils.metrics import compute_ppe, path_length, path_smoothness, safety_margin


def main():
    print("=" * 55)
    print("  DSPPP Framework — Quick Demo")
    print("  Dynamic Semantic Personalized Path Planning")
    print("=" * 55)

    # 1. Environment
    env = GridEnvironment(
        width=50, height=50,
        obstacle_density=0.20,
        n_dynamic=10,
        seed=42
    )
    start, goal = env.sample_start_goal()
    print(f"\nGrid: 50x50 | Static obstacles: 20% | Dynamic: 10")
    print(f"Start: {start}  →  Goal: {goal}")

    # 2. User preferences (IRL)
    learner = IRLPreferenceLearner(profile='balanced')
    alpha   = learner.alpha
    print(f"\nUser preference profile: balanced")
    print(f"Alpha weights: {np.round(alpha, 3)}")

    # 3. Dynamic penalty map
    dpm = DynamicPenaltyMap(graph_nodes=env.all_nodes)
    for obs in env.dynamic_obstacles:
        dpm.update_obstacle(obs['id'], np.array([obs['px'], obs['py']]))
    dpm._compute_penalties()
    penalty_map = dpm.get_map()

    obstacles = env.get_obstacle_measurements()

    # 4. Run DSPPP
    planner = DSPPPPlanner(env.graph, penalty_map, alpha)
    t0 = time.perf_counter()
    path, cost = planner.plan(start, goal, obstacles)
    t_ms = (time.perf_counter() - t0) * 1000

    # 5. Run Standard A* baseline
    astar = StandardAStar(env.graph)
    t0_a = time.perf_counter()
    path_a, nodes_a = astar.plan(start, goal)
    t_ms_a = (time.perf_counter() - t0_a) * 1000

    # 6. Compute metrics
    feature_dict = {}

    def metrics(p, t):
        if len(p) < 2:
            return None
        return {
            'time_ms':    round(t, 2),
            'length':     round(path_length(p, env.node_position), 2),
            'smoothness': round(path_smoothness(p, env.node_position), 2),
            'safety':     round(safety_margin(p, env.node_position, obstacles), 3),
            'ppe':        round(compute_ppe(p, env.node_position, alpha,
                                            feature_dict, obstacles, t), 6),
        }

    m_dsppp = metrics(path,   t_ms)
    m_astar = metrics(path_a, t_ms_a)

    # 7. Print comparison
    print("\n" + "-" * 55)
    print(f"{'Metric':<18} {'DSPPP':>12} {'Standard A*':>12}")
    print("-" * 55)
    if m_dsppp and m_astar:
        for key in m_dsppp:
            label = key.replace('_', ' ').capitalize()
            print(f"{label:<18} {str(m_dsppp[key]):>12} {str(m_astar[key]):>12}")

        print("-" * 55)
        speedup = m_astar['time_ms'] / m_dsppp['time_ms'] if m_dsppp['time_ms'] > 0 else 0
        print(f"  DSPPP speedup:    {speedup:.2f}x faster")
        if m_astar['smoothness'] > 0:
            smooth_imp = (m_astar['smoothness'] - m_dsppp['smoothness']) / m_astar['smoothness'] * 100
            print(f"  Smoothness gain:  {smooth_imp:.1f}% better")
        if m_astar['ppe'] > 0:
            ppe_imp = (m_dsppp['ppe'] - m_astar['ppe']) / m_astar['ppe'] * 100
            print(f"  PPE improvement:  {ppe_imp:.1f}% higher")
    else:
        print("  One or more planners failed to find a path.")

    print("\nDemo complete. Run experiments/run_experiment.py for full evaluation.")
    print("=" * 55)


if __name__ == '__main__':
    main()
