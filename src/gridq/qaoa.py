"""QAOA for the microgrid QUBO.

Pipeline:  Qubo -> Ising (SparsePauliOp) -> QAOAAnsatz(reps=p) -> transpile
           -> optimise (gamma, beta) with COBYLA on the exact statevector
              expectation (cheap at <= 12 qubits) or on sampled CVaR
           -> sample the final circuit with a SamplerV2 primitive
              (Aer ideal, Aer noise model, or IBM Runtime hardware)
           -> decode bitstrings, report best / most-likely / P(optimal).

Layer-wise warm start: the optimum at depth p seeds depth p+1 (INTERP rule,
Zhou et al., PRX 10, 021067, 2020), which avoids random restarts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from scipy.optimize import minimize

from .metrics import circuit_stats
from .qubo import Qubo, bitstring_to_array, energy_landscape


@dataclass
class QAOAResult:
    reps: int
    params: np.ndarray
    best_x: np.ndarray  # lowest-energy bitstring among samples
    best_energy: float
    top_x: np.ndarray  # most frequently sampled bitstring
    repaired_x: np.ndarray  # best dispatch after feasibility repair of every sample
    p_optimal: float  # probability mass on the ground state(s)
    p_near_optimal: float  # mass within 1% (of the energy range) of the optimum
    enrichment: float  # p_optimal / p_uniform: how much better than random guessing
    expected_energy: float  # <H> under the sampled distribution
    counts: dict[str, int]
    n_evals: int
    runtime_s: float
    circuit_stats: dict
    history: list[float] = field(default_factory=list)
    backend: str = "statevector"


def build_ansatz(qubo: Qubo, reps: int) -> tuple[QuantumCircuit, np.ndarray]:
    op, _, _ = qubo.to_ising(normalise=True)
    ansatz = QAOAAnsatz(op, reps=reps)
    return ansatz, energy_landscape(qubo)


def interp_init(params: np.ndarray) -> np.ndarray:
    """INTERP: map optimal depth-p (gammas, betas) to a depth-(p+1) start point."""
    p = len(params) // 2
    out = []
    for vec in (params[:p], params[p:]):  # QAOAAnsatz order: [betas..., gammas...]
        v = np.concatenate([[0.0], vec, [0.0]])
        out.append(np.array([(i / p) * v[i] + ((p - i) / p) * v[i + 1] for i in range(p + 1)]))
    return np.concatenate(out)


def _ordered(ansatz: QuantumCircuit) -> tuple[list, list]:
    betas = sorted([q for q in ansatz.parameters if "β" in q.name], key=lambda q: q.index)
    gammas = sorted([q for q in ansatz.parameters if "γ" in q.name], key=lambda q: q.index)
    return betas, gammas


def _bind(ansatz: QuantumCircuit, params: np.ndarray) -> QuantumCircuit:
    betas, gammas = _ordered(ansatz)
    p = len(betas)
    mapping = dict(zip(betas, params[:p])) | dict(zip(gammas, params[p:]))
    return ansatz.assign_parameters(mapping)


def ising_diagonal(qubo: Qubo) -> np.ndarray:
    """Diagonal of the normalised Ising operator, indexed by basis state k."""
    op, _, _ = qubo.to_ising(normalise=True)
    n = qubo.n
    k = np.arange(2**n, dtype=np.int64)
    bits = ((k[:, None] >> np.arange(n)) & 1).astype(np.int8)
    zsign = 1 - 2 * bits  # Z eigenvalue per qubit
    diag = np.zeros(2**n)
    for label, coeff in zip(op.paulis.to_labels(), op.coeffs):
        idx = [n - 1 - i for i, ch in enumerate(label) if ch == "Z"]
        diag += np.real(coeff) * (np.prod(zsign[:, idx], axis=1) if idx else 1.0)
    return diag


def fast_qaoa_probs(diag: np.ndarray, n: int, params: np.ndarray) -> np.ndarray:
    """Exact QAOA output distribution without building circuits.

    Matches QAOAAnsatz: |+>^n, then per layer exp(-i*gamma*H_C) followed by
    RX(2*beta) on every qubit. Parameter order is [betas..., gammas...].
    About 100x faster than Statevector(bound circuit) at 8-12 qubits.
    """
    p = len(params) // 2
    betas, gammas = params[:p], params[p:]
    psi = np.full(2**n, 2 ** (-n / 2), dtype=complex)
    for b, g in zip(betas, gammas):
        psi = psi * np.exp(-1j * g * diag)
        c, s = np.cos(b), -1j * np.sin(b)
        t = psi.reshape([2] * n)
        for q in range(n):
            ax = n - 1 - q  # qubit q is axis n-1-q in a C-ordered reshape
            a0 = np.take(t, 0, axis=ax)
            a1 = np.take(t, 1, axis=ax)
            t = np.stack([c * a0 + s * a1, s * a0 + c * a1], axis=ax)
        psi = t.reshape(-1)
    return np.abs(psi) ** 2


def grid_search_p1(diag: np.ndarray, n: int, e: np.ndarray, n_gamma: int = 41, n_beta: int = 33) -> np.ndarray:
    """Coarse grid over (beta, gamma) at p = 1; returns [beta, gamma] of the best point."""
    best_v, best = np.inf, np.array([0.3, 0.3])
    for gm in np.linspace(0.02, 2.0, n_gamma):
        for b in np.linspace(-np.pi / 2, np.pi / 2, n_beta):
            v = fast_qaoa_probs(diag, n, np.array([b, gm])) @ e
            if v < best_v:
                best_v, best = v, np.array([b, gm])
    return best


def _cvar(values: np.ndarray, probs: np.ndarray, alpha: float) -> float:
    """Mean of the lowest-energy alpha fraction of the distribution."""
    if alpha >= 1.0:
        return float(values @ probs)
    order = np.argsort(values)
    v, pr = values[order], probs[order]
    cum = np.cumsum(pr)
    k = int(np.searchsorted(cum, alpha))
    mass = np.concatenate([pr[:k], [alpha - (cum[k - 1] if k else 0.0)]])
    return float(v[: k + 1] @ mass / alpha)


def optimise_parameters(
    qubo: Qubo,
    reps: int,
    init: np.ndarray | None = None,
    alpha: float = 1.0,
    maxiter: int = 300,
    seed: int | None = 0,
) -> tuple[np.ndarray, list[float], int]:
    """Classical outer loop on the exact statevector (noise-free training).

    Training noise-free and then executing on noisy hardware is the standard
    approach for <= ~15 qubits; it saves hardware credits and queue time.
    """
    diag = ising_diagonal(qubo)
    # objective in [0, 1] units: 0 = ground state, 1 = worst state
    e = (diag - diag.min()) / (np.ptp(diag) or 1.0)
    if init is None:
        init = grid_search_p1(diag, qubo.n, e)
        if reps > 1:  # lift the p=1 point to depth `reps` by repeated INTERP
            for _ in range(reps - 1):
                init = interp_init(init)
    history: list[float] = []

    def f(params: np.ndarray) -> float:
        probs = fast_qaoa_probs(diag, qubo.n, params)
        val = _cvar(e, probs, alpha)
        history.append(val)
        return val

    res = minimize(f, init, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.3})
    return np.asarray(res.x), history, int(res.nfev)


def solve_qaoa(
    qubo: Qubo,
    reps: int = 2,
    sampler=None,
    pass_manager=None,
    shots: int = 4096,
    init: np.ndarray | None = None,
    alpha: float = 0.25,
    maxiter: int = 300,
    seed: int | None = 0,
    backend_name: str = "aer_ideal",
) -> QAOAResult:
    """Optimise QAOA angles, then sample the trained circuit on `sampler`.

    Args:
        sampler: any SamplerV2 (Aer, fake backend, or qiskit_ibm_runtime).
            Defaults to an ideal Aer sampler.
        pass_manager: transpiler for the target backend; defaults to a
            generic optimisation-level-3 pass manager.
        alpha: CVaR fraction for the objective (1.0 = plain expectation).
            Default 0.25: measured +40-50% enrichment vs alpha = 1 at n = 8-12.
    """
    from qiskit_aer.primitives import SamplerV2 as AerSampler

    t0 = time.perf_counter()
    params, history, n_evals = optimise_parameters(qubo, reps, init, alpha, maxiter, seed)
    ansatz, landscape = build_ansatz(qubo, reps)
    circ = _bind(ansatz, params)
    circ.measure_all()

    if pass_manager is None:
        pass_manager = generate_preset_pass_manager(
            optimization_level=3, basis_gates=["rz", "sx", "x", "cx"], seed_transpiler=seed or 0
        )
    isa = pass_manager.run(circ)
    sampler = sampler or AerSampler(seed=seed)
    counts = sampler.run([isa], shots=shots).result()[0].data.meas.get_counts()

    e_min = landscape.min()
    xs = {b: bitstring_to_array(b) for b in counts}
    es = {b: float(qubo.energy(xs[b])) for b in counts}
    best_b = min(es, key=es.get)
    top_b = max(counts, key=counts.get)
    total = sum(counts.values())
    p_opt = sum(c for b, c in counts.items() if np.isclose(es[b], e_min)) / total
    tol = 0.01 * (landscape.max() - e_min)
    p_near = sum(c for b, c in counts.items() if es[b] <= e_min + tol) / total
    exp_e = sum(es[b] * c for b, c in counts.items()) / total

    # Decode like an operator would: repair every sampled dispatch, keep the cheapest.
    from .classical import repair
    from .qubo import dispatch_objective

    repaired = {b: repair(qubo.grid, x) for b, x in xs.items()}
    rep_b = min(repaired, key=lambda b: float(dispatch_objective(qubo.grid, repaired[b], qubo.weights)))

    return QAOAResult(
        reps=reps,
        params=params,
        best_x=xs[best_b],
        best_energy=es[best_b],
        top_x=xs[top_b],
        repaired_x=repaired[rep_b],
        p_optimal=float(p_opt),
        p_near_optimal=float(p_near),
        enrichment=float(p_opt * np.isclose(landscape, e_min).size / max(1, np.isclose(landscape, e_min).sum())),
        expected_energy=float(exp_e),
        counts=dict(counts),
        n_evals=n_evals,
        runtime_s=time.perf_counter() - t0,
        circuit_stats=circuit_stats(isa),
        history=history,
        backend=backend_name,
    )


def depth_sweep(qubo: Qubo, max_reps: int = 4, **kw) -> list[QAOAResult]:
    """Run p = 1..max_reps with INTERP warm start between depths."""
    results, init = [], None
    for p in range(1, max_reps + 1):
        r = solve_qaoa(qubo, reps=p, init=init, **kw)
        results.append(r)
        init = interp_init(r.params)
    return results
