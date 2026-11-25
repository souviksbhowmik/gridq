"""QUBO formulation of microgrid load balancing.

    H(x) = A * (sum_i p_i x_i - D)^2          balance: cover the deficit
         + sum_i c_i x_i                      dispatch cost
         + B * sum_(i,j) in E  w_ij x_i x_j    line overload: w_ij > 0 only if
                                              p_i + p_j exceeds the line capacity
         - C * sum_i r_i x_i                  reward for supplying critical loads

All terms are normalised so that A, B, C are dimensionless and comparable
across instances. The penalty weights are calibrated so the brute-force optimum
of H is always a "good" dispatch (see tests/test_qubo.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization import QuadraticProgram

from .data import Microgrid


@dataclass
class QuboWeights:
    # A: calibrated over 36 instances (n = 6-10, density 0.2-0.9). Larger A
    # makes the QUBO optimum closer to feasible but flattens the cost signal
    # QAOA has to resolve; 20 is the compromise (see docs/formulation.md).
    balance: float = 20.0  # A
    congestion: float = 1.0  # B: penalty per unit of line overload
    priority: float = 0.04  # C: small bonus for supplying critical loads


@dataclass
class Qubo:
    """Dense upper-triangular QUBO: H(x) = x^T Q x + offset, Q[i, i] linear terms."""

    Q: np.ndarray
    offset: float
    grid: Microgrid
    weights: QuboWeights

    @property
    def n(self) -> int:
        return self.Q.shape[0]

    def energy(self, x: np.ndarray) -> np.ndarray:
        """Energy of one bitstring (shape (n,)) or many (shape (k, n))."""
        x = np.atleast_2d(x).astype(float)
        e = np.einsum("ki,ij,kj->k", x, self.Q, x) + self.offset
        return e if e.shape[0] > 1 else e[0]

    def to_quadratic_program(self) -> QuadraticProgram:
        qp = QuadraticProgram("microgrid_dispatch")
        for i in range(self.n):
            qp.binary_var(f"x{i}")
        linear = {f"x{i}": float(self.Q[i, i]) for i in range(self.n)}
        quadratic = {
            (f"x{i}", f"x{j}"): float(self.Q[i, j])
            for i in range(self.n)
            for j in range(i + 1, self.n)
            if self.Q[i, j] != 0
        }
        qp.minimize(constant=float(self.offset), linear=linear, quadratic=quadratic)
        return qp

    def to_ising(self, normalise: bool = True) -> tuple[SparsePauliOp, float, float]:
        """Ising Hamiltonian for QAOA.

        Returns (operator, offset, scale) such that
        H(x) = scale * (<z|operator|z> + offset), with qubit i <-> variable x_i.
        Normalising keeps the QAOA angle landscape on a similar scale for every
        instance, which makes parameter transfer between instances work.
        """
        op, offset = self.to_quadratic_program().to_ising()
        scale = 1.0
        if normalise:
            scale = float(np.max(np.abs(op.coeffs))) or 1.0
            op = (op / scale).simplify()
            offset = offset / scale
        return op, float(offset), scale


def build_qubo(grid: Microgrid, weights: QuboWeights | None = None) -> Qubo:
    w = weights or QuboWeights()
    n = grid.n
    p = grid.power / grid.deficit  # deficit becomes 1.0
    c = grid.cost / grid.cost.sum()  # costs sum to 1.0
    Q = np.zeros((n, n))

    # A * (sum p_i x_i - 1)^2 = A * [sum_i p_i^2 x_i + 2 sum_{i<j} p_i p_j x_i x_j - 2 sum_i p_i x_i + 1]
    for i in range(n):
        Q[i, i] += w.balance * (p[i] ** 2 - 2 * p[i])
        for j in range(i + 1, n):
            Q[i, j] += w.balance * 2 * p[i] * p[j]
    offset = w.balance * 1.0

    Q[np.diag_indices(n)] += c
    for (i, j), stress in grid.line_stress.items():
        a, b = min(i, j), max(i, j)
        Q[a, b] += w.congestion * stress
    Q[np.diag_indices(n)] -= w.priority * grid.priority
    return Qubo(Q=Q, offset=offset, grid=grid, weights=w)


def dispatch_objective(grid: Microgrid, X: np.ndarray, weights: QuboWeights | None = None) -> np.ndarray:
    """The *real* operator objective (cost + congestion - priority), no balance penalty.

    Used to compare solvers after feasibility repair: lower is better.
    """
    w = weights or QuboWeights()
    X = np.atleast_2d(X).astype(float)
    c = grid.cost / grid.cost.sum()
    cong = np.zeros(X.shape[0])
    for (i, j), s in grid.line_stress.items():
        cong += s * X[:, i] * X[:, j]
    out = X @ c + w.congestion * cong - w.priority * (X @ grid.priority)
    return out if out.shape[0] > 1 else out[0]


def all_bitstrings(n: int) -> np.ndarray:
    """All 2^n bitstrings, row k has x_i = bit i of k (Qiskit little-endian order)."""
    k = np.arange(2**n, dtype=np.int64)
    return ((k[:, None] >> np.arange(n)) & 1).astype(np.int8)


def energy_landscape(qubo: Qubo) -> np.ndarray:
    """H(x) for every basis state index k (matches Statevector / counts ordering)."""
    return qubo.energy(all_bitstrings(qubo.n))


def bitstring_to_array(bits: str) -> np.ndarray:
    """Qiskit counts key (rightmost char = qubit 0) -> array x with x[i] = qubit i."""
    return np.array([int(b) for b in reversed(bits)], dtype=np.int8)
