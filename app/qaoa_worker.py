"""Run QAOA in a separate process for the Streamlit dashboard.

Qiskit's native (Rust) extension crashes with a segmentation fault when it is
called from inside the Streamlit server process on macOS (seen in
SparsePauliOp.simplify). The same code is fine in a normal Python process. So
the dashboard runs each QAOA solve here, in a fresh "spawn" process, and gets
plain numbers back. If the worker crashes, the dashboard shows an error
instead of shutting down.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np


def _solve(n, power, cost, deficit, reps, shots, noisy, out):
    try:
        from gridq import build_qubo, solve_qaoa
        from gridq.data import Microgrid

        grid = Microgrid(n=n, power=np.asarray(power), cost=np.asarray(cost), deficit=deficit)
        qubo = build_qubo(grid)
        sampler = pm = None
        if noisy:
            from gridq.hardware import backend_pass_manager, fake_backend, noisy_sampler

            fb = fake_backend()
            sampler, pm = noisy_sampler(fb, seed=0), backend_pass_manager(fb)
        r = solve_qaoa(qubo, reps=reps, shots=shots, sampler=sampler, pass_manager=pm)
        out.put({
            "ok": True, "x": r.repaired_x.tolist(), "enrichment": r.enrichment, "p_optimal": r.p_optimal,
            "evals": r.n_evals, "two_q_depth": r.circuit_stats["two_qubit_depth"],
            "two_q_gates": r.circuit_stats["two_qubit_gates"], "counts": r.counts,
        })
    except Exception as e:  # report Python errors back to the dashboard
        out.put({"ok": False, "error": f"{type(e).__name__}: {e}"})


def solve_qaoa_isolated(grid, reps: int, shots: int, noisy: bool, timeout: float = 300.0) -> dict:
    """Solve one dispatch instance with QAOA in a child process; returns a result dict."""
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    p = ctx.Process(target=_solve, args=(grid.n, grid.power.tolist(), grid.cost.tolist(), grid.deficit,
                                          reps, shots, noisy, out), daemon=True)
    p.start()
    try:
        res = out.get(timeout=timeout)
    except Exception:
        res = {"ok": False, "error": f"QAOA worker stopped (exit code {p.exitcode})"}
    p.join(timeout=5)
    if p.is_alive():
        p.kill()
    return res
