"""GridQ: QAOA load balancing for rural microgrids (QFF 2026, Climate track)."""

from .classical import (
    brute_force, constrained_optimum, greedy_descent, greedy_operator, random_sampling, repair,
    simulated_annealing,
)
from .data import Microgrid, daily_profile, make_microgrid
from .metrics import approximation_ratio, cost_saved_pct, dispatch_metrics
from .qaoa import depth_sweep, solve_qaoa
from .qubo import Qubo, QuboWeights, build_qubo, dispatch_objective, problem_graph_density, sparsify_qubo

__all__ = [
    "Microgrid", "make_microgrid", "daily_profile",
    "Qubo", "QuboWeights", "build_qubo", "dispatch_objective", "sparsify_qubo", "problem_graph_density",
    "brute_force", "constrained_optimum", "repair", "greedy_operator", "greedy_descent", "simulated_annealing", "random_sampling",
    "solve_qaoa", "depth_sweep",
    "approximation_ratio", "dispatch_metrics", "cost_saved_pct",
]
__version__ = "0.1.0"
