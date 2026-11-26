"""Solution-quality and deployment metrics."""

from __future__ import annotations

import numpy as np

from .data import Microgrid


def approximation_ratio(energy: float, e_min: float, e_max: float) -> float:
    """1.0 = optimal, 0.0 = worst possible bitstring. Works for negative energies."""
    if np.isclose(e_max, e_min):
        return 1.0
    return float((e_max - energy) / (e_max - e_min))


def dispatch_metrics(grid: Microgrid, x: np.ndarray) -> dict:
    """Operator-facing numbers for a dispatch decision x."""
    x = np.asarray(x, dtype=float)
    supplied = float(grid.power @ x)
    served = min(supplied, grid.deficit)
    return {
        "supplied_kw": round(supplied, 2),
        "deficit_kw": grid.deficit,
        "demand_met_pct": round(100 * served / grid.deficit, 1),
        "surplus_kw": round(max(0.0, supplied - grid.deficit), 2),
        "cost_inr": round(float(grid.cost @ x), 1),
        "assets_on": int(x.sum()),
    }


def naive_policy_cost(grid: Microgrid) -> float:
    """Baseline "deployment" cost: cover the deficit with the dearest assets.

    Represents today's default in many villages: run diesel first because it is
    reliable and always available. Used for the "cost saved" headline number.
    """
    order = np.argsort(-grid.cost / grid.power)
    supplied, cost = 0.0, 0.0
    for i in order:
        if supplied >= grid.deficit:
            break
        supplied += grid.power[i]
        cost += grid.cost[i]
    return float(cost)


def cost_saved_pct(grid: Microgrid, x: np.ndarray) -> float:
    base = naive_policy_cost(grid)
    return round(100 * (base - float(grid.cost @ np.asarray(x))) / base, 1) if base else 0.0


def circuit_stats(circuit) -> dict:
    ops = circuit.count_ops()
    two_q = sum(v for k, v in ops.items() if k in {"cx", "cz", "ecr", "rzz", "swap"})
    return {
        "depth": circuit.depth(),
        "two_qubit_gates": int(two_q),
        "two_qubit_depth": circuit.depth(lambda inst: inst.operation.num_qubits == 2),
        "num_qubits": circuit.num_qubits,
    }
