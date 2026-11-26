"""Classical baselines: brute force, greedy and simulated annealing."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .qubo import Qubo, all_bitstrings, dispatch_objective


@dataclass
class SolverResult:
    name: str
    x: np.ndarray
    energy: float
    runtime_s: float
    extra: dict | None = None


def repair(grid, x: np.ndarray) -> np.ndarray:
    """Make any dispatch feasible (supply >= deficit), then trim waste.

    Applied identically to every solver's output, so comparisons stay fair.
    1. While short: switch on the off-asset with the lowest cost per kW.
    2. Then: switch off on-assets (most expensive first) if still feasible
       and the objective improves.
    """
    x = np.asarray(x, dtype=np.int8).copy()
    ratio = grid.cost / grid.power
    while grid.power @ x < grid.deficit - 1e-9:
        off = np.flatnonzero(x == 0)
        if off.size == 0:
            break
        x[off[np.argmin(ratio[off])]] = 1
    for i in np.argsort(-grid.cost):
        if x[i] and grid.power @ x - grid.power[i] >= grid.deficit - 1e-9:
            y = x.copy()
            y[i] = 0
            if dispatch_objective(grid, y) < dispatch_objective(grid, x):  # only trim if it helps
                x = y
    return x


def constrained_optimum(qubo: Qubo) -> SolverResult:
    """Exact ground truth for the *real* problem: min objective s.t. supply >= deficit."""
    t0 = time.perf_counter()
    g = qubo.grid
    X = all_bitstrings(g.n)
    obj = dispatch_objective(g, X, qubo.weights)
    obj = np.where(X @ g.power >= g.deficit - 1e-9, obj, np.inf)
    k = int(np.argmin(obj))
    return SolverResult("exact_constrained", X[k].copy(), float(qubo.energy(X[k])), time.perf_counter() - t0)


def brute_force(qubo: Qubo) -> SolverResult:
    """Exact optimum by enumerating all 2^n states (fine up to n ~ 20)."""
    t0 = time.perf_counter()
    X = all_bitstrings(qubo.n)
    E = qubo.energy(X)
    k = int(np.argmin(E))
    return SolverResult(
        "brute_force", X[k].copy(), float(E[k]), time.perf_counter() - t0,
        extra={"e_max": float(E.max())},
    )


def greedy_operator(qubo: Qubo) -> SolverResult:
    """What a rural grid operator does by hand today.

    Switch on assets in order of cheapest cost per kW until the deficit is
    covered.
    """
    t0 = time.perf_counter()
    g = qubo.grid
    order = np.argsort(g.cost / g.power, kind="stable")
    x = np.zeros(g.n, dtype=np.int8)
    supplied = 0.0
    for i in order:
        if supplied >= g.deficit:
            break
        x[i] = 1
        supplied += g.power[i]
    return SolverResult("greedy", x, float(qubo.energy(x)), time.perf_counter() - t0)


def greedy_descent(qubo: Qubo) -> SolverResult:
    """Greedy on the QUBO itself: flip the single best bit until no improvement."""
    t0 = time.perf_counter()
    x = np.zeros(qubo.n, dtype=np.int8)
    e = float(qubo.energy(x))
    while True:
        cand = np.repeat(x[None, :], qubo.n, axis=0)
        idx = np.arange(qubo.n)
        cand[idx, idx] ^= 1
        ec = qubo.energy(cand)
        k = int(np.argmin(ec))
        if ec[k] >= e - 1e-12:
            break
        x, e = cand[k], float(ec[k])
    return SolverResult("greedy_descent", x, e, time.perf_counter() - t0)


def simulated_annealing(
    qubo: Qubo,
    sweeps: int = 1000,
    restarts: int = 10,
    t_start: float = 2.0,
    t_end: float = 0.01,
    seed: int | None = 0,
) -> SolverResult:
    """Single-bit-flip Metropolis annealing with a geometric schedule."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = qubo.n
    S = qubo.Q + qubo.Q.T  # symmetric couplings, diagonal = 2 * linear
    lin = np.diag(qubo.Q).copy()
    temps = np.geomspace(t_start, t_end, sweeps)
    best_x, best_e = None, np.inf
    for _ in range(restarts):
        x = rng.integers(0, 2, n).astype(float)
        for T in temps:
            for i in rng.permutation(n):
                # dE of flipping x_i: (1 - 2 x_i) * (lin_i + sum_{j != i} S_ij x_j)
                field = lin[i] + S[i] @ x - S[i, i] * x[i]
                d_e = (1 - 2 * x[i]) * field
                if d_e <= 0 or rng.random() < np.exp(-d_e / T):
                    x[i] = 1 - x[i]
        e = float(qubo.energy(x))
        if e < best_e:
            best_x, best_e = x.astype(np.int8), e
    return SolverResult(
        "simulated_annealing", best_x, best_e, time.perf_counter() - t0,
        extra={"sweeps": sweeps, "restarts": restarts},
    )


def random_sampling(qubo: Qubo, shots: int = 4096, seed: int | None = 0) -> SolverResult:
    """Control baseline: uniform random bitstrings, same shot budget and same
    repair-and-pick-best decoding as QAOA. If QAOA cannot beat this, the
    quantum circuit is not contributing anything."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    g = qubo.grid
    X = np.unique(rng.integers(0, 2, (shots, g.n)).astype(np.int8), axis=0)
    R = np.array([repair(g, x) for x in X])
    obj = dispatch_objective(g, R, qubo.weights)
    k = int(np.argmin(obj))
    return SolverResult("random_sampling", R[k], float(qubo.energy(R[k])), time.perf_counter() - t0,
                        extra={"distinct_states": len(X)})
