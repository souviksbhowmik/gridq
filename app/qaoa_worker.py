"""Run every Qiskit task for the Streamlit dashboard in a separate process.

Qiskit's native (Rust) extension crashes with a segmentation fault when it is
called from inside the Streamlit server process on macOS (seen in
SparsePauliOp.simplify). The same code is fine in a normal Python process. So
the dashboard runs each Qiskit task here, in a fresh Python subprocess, and gets
plain numbers back as JSON. If the worker crashes, the dashboard shows an error
instead of shutting down.

Tasks
  qaoa       train + sample QAOA on the ideal or noisy simulator
  hardware   train on the simulator, then submit to a real IBM QPU (uses credits)
  score      re-score a saved results/hardware_*.json for the pitch page
  backend    look up which QPU would be used (no job submitted)
"""

from __future__ import annotations

import numpy as np


def _grid(n, power, cost, deficit):
    from gridq.data import Microgrid

    return Microgrid(n=n, power=np.asarray(power), cost=np.asarray(cost), deficit=deficit)


def _task_qaoa(n, power, cost, deficit, reps, shots, noisy):
    from gridq import build_qubo, solve_qaoa

    qubo = build_qubo(_grid(n, power, cost, deficit))
    sampler = pm = None
    if noisy:
        from gridq.hardware import backend_pass_manager, fake_backend, noisy_sampler

        fb = fake_backend()
        sampler, pm = noisy_sampler(fb, seed=0), backend_pass_manager(fb)
    r = solve_qaoa(qubo, reps=reps, shots=shots, sampler=sampler, pass_manager=pm)
    return {"x": r.repaired_x.tolist(), "enrichment": r.enrichment, "p_optimal": r.p_optimal,
            "evals": r.n_evals, "two_q_depth": r.circuit_stats["two_qubit_depth"],
            "two_q_gates": r.circuit_stats["two_qubit_gates"], "counts": r.counts}


def _task_hardware(n, power, cost, deficit, reps, shots, mitigate):
    """Same pipeline as experiments/run_hardware.py, for one dashboard instance."""
    import time

    from gridq import build_qubo, repair
    from gridq.hardware import backend_pass_manager, get_backend, hardware_sampler
    from gridq.metrics import circuit_stats
    from gridq.qaoa import _bind, build_ansatz, optimise_parameters
    from gridq.qubo import bitstring_to_array, dispatch_objective, energy_landscape

    grid = _grid(n, power, cost, deficit)
    qubo = build_qubo(grid)
    params, _, nev = optimise_parameters(qubo, reps, alpha=0.25)  # trained on the simulator: no credits
    ansatz, _ = build_ansatz(qubo, reps)
    circ = _bind(ansatz, params)
    circ.measure_all()
    backend = get_backend(min_qubits=n)
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
            "counts": counts, "backend": backend.name, "job_id": job.job_id(), "queue_and_run_s": wall}


def _task_score(path):
    from gridq.hardware import score_hardware_run

    return score_hardware_run(path)


def _task_backend():
    from gridq.hardware import get_backend

    b = get_backend(min_qubits=6)
    status = b.status()
    return {"backend": b.name, "qubits": b.num_qubits, "pending_jobs": getattr(status, "pending_jobs", None),
            "operational": getattr(status, "operational", None)}


TASKS = {"qaoa": _task_qaoa, "hardware": _task_hardware, "score": _task_score, "backend": _task_backend}


def run_isolated(task: str, *args, timeout: float = 300.0) -> dict:
    """Run one task in a fresh Python subprocess; returns {"ok": bool, ...} and never raises.

    A plain subprocess (not multiprocessing) is used so the child imports only
    this module. multiprocessing's "spawn" re-executes the parent's __main__,
    which for Streamlit is the whole dashboard script.
    """
    import json
    import subprocess
    import sys
    from pathlib import Path

    try:
        proc = subprocess.run(
            [sys.executable, "-I", str(Path(__file__).resolve()), task],
            input=json.dumps(list(args)), capture_output=True, text=True, timeout=timeout,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"worker timed out after {timeout:.0f} s"}
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(_MARK)]
    if not lines:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        return {"ok": False, "error": f"worker exited with code {proc.returncode}: {' | '.join(tail)}"}
    return json.loads(lines[-1][len(_MARK):])


_MARK = "@@GRIDQ_RESULT@@"


def _main():
    import json
    import sys
    import warnings

    warnings.filterwarnings("ignore")
    task, args = sys.argv[1], json.loads(sys.stdin.read())
    try:
        res = {"ok": True, **TASKS[task](*args)}
    except Exception as e:  # report Python errors back to the dashboard
        res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print(_MARK + json.dumps(res, default=_jsonable), flush=True)


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _grid_args(grid):
    return grid.n, grid.power.tolist(), grid.cost.tolist(), grid.deficit


def solve_qaoa_isolated(grid, reps: int, shots: int, noisy: bool, timeout: float = 300.0) -> dict:
    return run_isolated("qaoa", *_grid_args(grid), reps, shots, noisy, timeout=timeout)


def solve_hardware_isolated(grid, reps: int, shots: int, mitigate: bool, timeout: float = 1800.0) -> dict:
    return run_isolated("hardware", *_grid_args(grid), reps, shots, mitigate, timeout=timeout)


def score_saved_run_isolated(path: str) -> dict:
    return run_isolated("score", str(path), timeout=300.0)


def hardware_backend_isolated() -> dict:
    return run_isolated("backend", timeout=120.0)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    _main()
