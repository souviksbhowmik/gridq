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
            "params": r.params.tolist(), "evals": r.n_evals, "two_q_depth": r.circuit_stats["two_qubit_depth"],
            "two_q_gates": r.circuit_stats["two_qubit_gates"], "counts": r.counts}


def _task_hardware(n, power, cost, deficit, reps, shots, mitigate):
    """Same pipeline as experiments/run_hardware.py, for one dashboard instance."""
    from gridq.hardware import run_qaoa_on_hardware

    r = run_qaoa_on_hardware(_grid(n, power, cost, deficit), reps=reps, shots=shots, mitigate=mitigate)
    r.pop("params", None)
    return r


def _task_circuit(n, power, cost, deficit, params, png_path):
    """Draw the trained QAOA circuit (gate by gate) to a PNG; return gate counts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gridq import build_qubo
    from gridq.qaoa import readable_circuit

    circ = readable_circuit(build_qubo(_grid(n, power, cost, deficit)), np.asarray(params))
    fig = circ.draw("mpl", fold=30, scale=0.6)
    fig.savefig(png_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    ops = circ.count_ops()
    return {"png": png_path, "rzz": ops.get("rzz", 0), "rz": ops.get("rz", 0), "rx": ops.get("rx", 0),
            "depth": circ.depth(), "reps": len(params) // 2}


def _task_score(path):
    from gridq.hardware import score_hardware_run

    return score_hardware_run(path)


def _task_backend():
    from gridq.hardware import get_backend

    b = get_backend(min_qubits=6)
    status = b.status()
    return {"backend": b.name, "qubits": b.num_qubits, "pending_jobs": getattr(status, "pending_jobs", None),
            "operational": getattr(status, "operational", None)}


TASKS = {"qaoa": _task_qaoa, "hardware": _task_hardware, "score": _task_score, "backend": _task_backend,
         "circuit": _task_circuit}


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


def draw_circuit_isolated(grid, params, png_path: str) -> dict:
    return run_isolated("circuit", *_grid_args(grid), list(map(float, params)), str(png_path), timeout=180.0)
