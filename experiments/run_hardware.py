"""Run one trained QAOA instance on real IBM Quantum hardware.

    python experiments/run_hardware.py --dry-run          # transpile only, no credits used
    python experiments/run_hardware.py --n 6 --reps 2     # submit (uses credits)

Workflow (as the rules ask: simulator first, then hardware):
  1. Train (gamma, beta) noise-free on the exact simulator (no credits).
  2. Transpile for the chosen backend (optimisation level 3) and report depth.
  3. Submit 2 jobs in one session-less batch: unmitigated and mitigated
     (dynamical decoupling + gate/measurement twirling).
  4. Save counts, job IDs and metrics to results/hardware_*.json.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

from gridq import build_qubo, constrained_optimum, make_microgrid, repair
from gridq.hardware import backend_pass_manager, get_backend, hardware_sampler
from gridq.metrics import circuit_stats, dispatch_metrics
from gridq.qaoa import _bind, build_ansatz, optimise_parameters
from gridq.qubo import bitstring_to_array, dispatch_objective, energy_landscape

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[1] / "results"


def summarise(qubo, counts):
    g = qubo.grid
    land = energy_landscape(qubo)
    e_min = land.min()
    total = sum(counts.values())
    es = {b: float(qubo.energy(bitstring_to_array(b))) for b in counts}
    p_opt = sum(c for b, c in counts.items() if np.isclose(es[b], e_min)) / total
    n_opt = int(np.isclose(land, e_min).sum())
    rep = {b: repair(g, bitstring_to_array(b)) for b in counts}
    best = min(rep, key=lambda b: float(dispatch_objective(g, rep[b], qubo.weights)))
    exact = constrained_optimum(qubo).x
    return {
        "shots": total,
        "p_optimal": p_opt,
        "enrichment": p_opt * land.size / n_opt,
        "expected_energy": sum(es[b] * c for b, c in counts.items()) / total,
        "found_optimal_dispatch": bool(np.array_equal(rep[best], exact)),
        "dispatch": dispatch_metrics(g, rep[best]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--dry-run", action="store_true", help="transpile and report, do not submit")
    args = ap.parse_args()

    qubo = build_qubo(make_microgrid(args.n, seed=args.seed))
    params, _, _ = optimise_parameters(qubo, args.reps, alpha=0.25)
    ansatz, _ = build_ansatz(qubo, args.reps)
    circ = _bind(ansatz, params)
    circ.measure_all()

    backend = get_backend(min_qubits=args.n)
    isa = backend_pass_manager(backend, seed=args.seed).run(circ)
    stats = circuit_stats(isa)
    print(f"backend={backend.name}  transpiled: {stats}")
    if args.dry_run:
        print("dry run: nothing submitted")
        return

    jobs = {}
    for label, mitigate in (("unmitigated", False), ("mitigated", True)):
        sampler = hardware_sampler(backend, shots=args.shots, mitigate=mitigate)
        job = sampler.run([isa])
        jobs[label] = job
        print(f"submitted {label}: job_id={job.job_id()}")

    out = {"backend": backend.name, "args": vars(args), "params": params.tolist(),
           "circuit": stats, "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": {}}
    for label, job in jobs.items():
        print(f"waiting for {label} ({job.job_id()}) ...")
        counts = job.result()[0].data.meas.get_counts()
        out["runs"][label] = {"job_id": job.job_id(), **summarise(qubo, counts), "counts": counts}
        print(label, {k: v for k, v in out["runs"][label].items() if k != "counts"})

    path = OUT / f"hardware_{backend.name}_n{args.n}_p{args.reps}_s{args.seed}.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
