"""IBM Quantum execution: credentials from .env, noise-model and real-hardware samplers.

Credentials are read from environment variables (loaded from a git-ignored
.env file). Nothing secret is ever written to disk by this module.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

ROOT = Path(__file__).resolve().parents[2]


def _service():
    from qiskit_ibm_runtime import QiskitRuntimeService

    load_dotenv(ROOT / ".env")
    token = os.getenv("QISKIT_IBM_TOKEN", "").strip()
    if not token:
        raise RuntimeError("QISKIT_IBM_TOKEN is empty: fill in .env (see .env.example)")
    return QiskitRuntimeService(
        channel=os.getenv("QISKIT_IBM_CHANNEL", "ibm_quantum_platform").strip(),
        token=token,
        instance=os.getenv("QISKIT_IBM_INSTANCE", "").strip() or None,
    )


def get_backend(min_qubits: int = 12):
    """Backend named in IBM_BACKEND, or the least-busy operational one."""
    service = _service()
    name = os.getenv("IBM_BACKEND", "").strip()
    if name:
        return service.backend(name)
    return service.least_busy(operational=True, simulator=False, min_num_qubits=min_qubits)


def hardware_sampler(backend, shots: int = 4096, mitigate: bool = True):
    """Runtime SamplerV2 with dynamical decoupling + gate twirling.

    For a sampler, twirling + DD are the standard mitigation knobs; readout
    errors are additionally suppressed by measurement twirling.
    """
    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = shots
    if mitigate:
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XpXm"
        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.enable_measure = True
        sampler.options.twirling.num_randomizations = "auto"
    return sampler


def noisy_sampler(backend=None, seed: int = 0):
    """Aer sampler with the noise model of `backend`.

    If no backend is given (or no credentials), uses a bundled fake Heron
    backend snapshot so the experiment runs fully offline.
    """
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2 as AerSampler

    if backend is None:
        backend = fake_backend()
    sim = AerSimulator.from_backend(backend, seed_simulator=seed)
    return AerSampler.from_backend(sim, seed=seed) if hasattr(AerSampler, "from_backend") else AerSampler(
        options={"backend_options": {"noise_model": sim.options.noise_model}, "run_options": {"seed": seed}}
    )


def fake_backend():
    """Offline snapshot of a real IBM device (Heron r2 'ibm_fez' if available)."""
    from qiskit_ibm_runtime import fake_provider

    for name in ("FakeFez", "FakeTorino", "FakeSherbrooke"):
        if hasattr(fake_provider, name):
            return getattr(fake_provider, name)()
    raise RuntimeError("No fake backend available in qiskit_ibm_runtime.fake_provider")


def backend_pass_manager(backend, seed: int = 0):
    return generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=seed)


# ---------------------------------------------------------------------------
# Saved hardware runs (results/hardware_*.json): scoring for reports/dashboard
# ---------------------------------------------------------------------------

def counts_to_probs(counts: dict, n: int):
    """Qiskit counts -> probability vector indexed by basis state (bit i = provider i)."""
    import numpy as np

    p = np.zeros(2**n)
    for bits, c in counts.items():
        p[int(bits, 2)] += c
    return p / p.sum()


def score_hardware_run(path, noise_model: bool = True, seed: int = 0) -> dict:
    """Re-score a saved hardware JSON against the current model.

    Returns one row per setting (ideal, noise model, each hardware run,
    random guessing) with P(optimal), enrichment, <H>, fidelity to the ideal
    distribution, and the dispatch each run decodes to. The ideal and
    noise-model rows use the same trained angles that were sent to the QPU.
    """
    import json

    import numpy as np

    from .classical import constrained_optimum, repair
    from .data import make_microgrid
    from .metrics import dispatch_metrics
    from .qaoa import _bind, build_ansatz, fast_qaoa_probs, ising_diagonal
    from .qubo import build_qubo, dispatch_objective, energy_landscape

    d = json.loads(Path(path).read_text())
    a = d["args"]
    n, reps = a["n"], a["reps"]
    grid = make_microgrid(n, seed=a["seed"])
    qubo = build_qubo(grid)
    land = energy_landscape(qubo)
    opt = np.isclose(land, land.min())
    exact = constrained_optimum(qubo).x
    params = np.asarray(d["params"])
    ideal = fast_qaoa_probs(ising_diagonal(qubo), n, params)
    shots = next(iter(d["runs"].values()))["shots"]

    dists = {"Ideal simulator": ideal}
    if noise_model:
        ansatz, _ = build_ansatz(qubo, reps)
        circ = _bind(ansatz, params)
        circ.measure_all()
        fb = fake_backend()
        res = noisy_sampler(fb, seed=seed).run([backend_pass_manager(fb, seed).run(circ)], shots=shots).result()
        dists[f"Noise model ({fb.name})"] = counts_to_probs(res[0].data.meas.get_counts(), n)
    for label, run in d["runs"].items():
        dists[f"{d['backend']} · {label}"] = counts_to_probs(run["counts"], n)
    dists["Random guessing"] = np.full(2**n, 2.0**-n)

    rows = []
    for name, p in dists.items():
        po = float(p[opt].sum())
        top = np.argsort(-p)[:256]
        bits = ((top[:, None] >> np.arange(n)) & 1).astype(np.int8)
        cand = np.array([repair(grid, b) for b in bits])
        best = cand[np.argmin(dispatch_objective(grid, cand))]
        rows.append({
            "setting": name,
            "kind": "hardware" if name.startswith(d["backend"]) else ("baseline" if "Random" in name else "simulator"),
            "p_optimal": po,
            "p_optimal_err": float(np.sqrt(po * (1 - po) / shots)) if "Random" not in name else 0.0,
            "enrichment": po * p.size / opt.sum(),
            "expected_energy": float(p @ land),
            "fidelity_to_ideal": float(np.sum(np.sqrt(p * ideal)) ** 2),
            "found_optimal": bool(np.array_equal(best, exact)),
            "providers_on": [int(i) for i in np.flatnonzero(best)],
            "supplied_kw": float(grid.power @ best),
            "cost_inr": float(grid.cost @ best),
            "job_id": d["runs"][name.split(" · ")[1]]["job_id"] if " · " in name else "",
        })
    return {
        "backend": d["backend"], "submitted_at": d.get("submitted_at", ""), "n": n, "reps": reps,
        "seed": a["seed"], "shots": shots, "circuit": d["circuit"], "rows": rows,
        "deficit_kw": grid.deficit, "optimal_dispatch": dispatch_metrics(grid, exact),
        "optimal_providers": [int(i) for i in np.flatnonzero(exact)],
        "probs": {k: v.tolist() for k, v in dists.items()}, "optimal_states": np.flatnonzero(opt).tolist(),
        "params": params.tolist(), "power": grid.power.tolist(), "cost": grid.cost.tolist(),
    }


def run_qaoa_on_hardware(grid, reps: int = 2, shots: int = 4096, mitigate: bool = False) -> dict:
    """Train QAOA on the simulator (free), then sample the circuit once on a real IBM QPU.

    Uses the backend from .env (IBM_BACKEND, or the least-busy QPU). Submits
    one job and waits for it. Returns the decoded dispatch and quality numbers.
    """
    import time

    import numpy as np

    from .classical import repair
    from .metrics import circuit_stats
    from .qaoa import _bind, build_ansatz, optimise_parameters
    from .qubo import bitstring_to_array, build_qubo, dispatch_objective, energy_landscape

    qubo = build_qubo(grid)
    params, _, nev = optimise_parameters(qubo, reps, alpha=0.25)
    ansatz, _ = build_ansatz(qubo, reps)
    circ = _bind(ansatz, params)
    circ.measure_all()
    backend = get_backend(min_qubits=grid.n)
    isa = backend_pass_manager(backend).run(circ)
    t0 = time.perf_counter()
    job = hardware_sampler(backend, shots=shots, mitigate=mitigate).run([isa])
    counts = job.result()[0].data.meas.get_counts()
    wall = time.perf_counter() - t0
    land = energy_landscape(qubo)
    opt = np.isclose(land, land.min())
    total = sum(counts.values())
    p_opt = sum(c for b, c in counts.items() if opt[int(b, 2)]) / total
    rep = {b: repair(grid, bitstring_to_array(b)) for b in counts}
    best = min(rep, key=lambda b: float(dispatch_objective(grid, rep[b])))
    stats = circuit_stats(isa)
    return {"x": rep[best].tolist(), "enrichment": p_opt * land.size / opt.sum(), "p_optimal": p_opt,
            "evals": nev, "two_q_depth": stats["two_qubit_depth"], "two_q_gates": stats["two_qubit_gates"],
            "counts": counts, "backend": backend.name, "job_id": job.job_id(), "queue_and_run_s": wall,
            "params": params.tolist()}
