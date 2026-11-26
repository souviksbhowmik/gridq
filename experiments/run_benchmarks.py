"""Run the benchmark experiments and write CSVs + plots to results/.

    python experiments/run_benchmarks.py            # all experiments (~ a few minutes)
    python experiments/run_benchmarks.py --quick    # fewer seeds, for a smoke test
    python experiments/run_benchmarks.py --only density

Experiments
  correctness  all solvers vs exact constrained optimum, n = 6/8/10, many seeds
  depth        QAOA quality vs p = 1..5 (INTERP warm start)
  density      QAOA convergence, quality and 2q depth vs problem-graph density (sparsified QUBO)
  scaling      runtime and quality vs n = 6..12, all solvers
  noise        ideal vs fake-Heron noise model, p = 1..3
  day          24-h dispatch of data/villages/village_n8_s0 (village demand - solar - wind)
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gridq import (
    build_qubo, constrained_optimum, cost_saved_pct, daily_profile, depth_sweep, dispatch_metrics,
    dispatch_objective, greedy_descent, greedy_operator, make_microgrid, random_sampling, repair,
    simulated_annealing,
    solve_qaoa,
)
from gridq.metrics import naive_policy_cost
from gridq.qubo import all_bitstrings

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)


def quality(grid, x, weights) -> float:
    """Normalised optimality of a *feasible* dispatch: 1 = exact optimum, 0 = worst feasible."""
    X = all_bitstrings(grid.n)
    obj = dispatch_objective(grid, X, weights)
    feas = X @ grid.power >= grid.deficit - 1e-9
    lo, hi = obj[feas].min(), obj[feas].max()
    v = float(dispatch_objective(grid, x, weights))
    return 1.0 if np.isclose(hi, lo) else float((hi - v) / (hi - lo))


def solve_all(q, reps=3, include_qaoa=True, shots=256):
    g = q.grid
    rows = []
    t = time.perf_counter()
    exact = constrained_optimum(q)
    solvers = [
        ("exact", exact.x, exact.runtime_s),
        ("greedy_operator", greedy_operator(q).x, None),
        ("greedy_descent", None, None),
        ("simulated_annealing", None, None),
    ]
    gd = greedy_descent(q)
    sa = simulated_annealing(q, sweeps=500, restarts=5)
    solvers[2] = ("greedy_descent", gd.x, gd.runtime_s)
    solvers[3] = ("simulated_annealing", sa.x, sa.runtime_s)
    go = greedy_operator(q)
    solvers[1] = ("greedy_operator", go.x, go.runtime_s)
    rs = random_sampling(q, shots=shots, seed=g.seed)
    solvers.append(("random_sampling", rs.x, rs.runtime_s))
    enrich = {}
    if include_qaoa:
        r = solve_qaoa(q, reps=reps, shots=shots)
        solvers.append((f"qaoa_p{reps}", r.repaired_x, r.runtime_s))
        enrich[f"qaoa_p{reps}"] = r.enrichment
    for name, x, rt in solvers:
        x = repair(g, x)
        m = dispatch_metrics(g, x)
        rows.append(dict(
            solver=name, n=g.n, seed=g.seed,
            quality=quality(g, x, q.weights), optimal=bool(np.array_equal(x, exact.x)),
            cost_inr=m["cost_inr"], demand_met_pct=m["demand_met_pct"],
            cost_saved_pct=cost_saved_pct(g, x), runtime_s=rt, enrichment=enrich.get(name, np.nan),
        ))
    return rows


def exp_correctness(seeds):
    rows = []
    for n in (6, 8, 10):
        for s in seeds:
            rows += solve_all(build_qubo(make_microgrid(n, seed=s)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "correctness.csv", index=False)
    summ = df.groupby(["solver", "n"]).agg(
        quality=("quality", "mean"), optimal_rate=("optimal", "mean"),
        cost_saved_pct=("cost_saved_pct", "mean"), runtime_ms=("runtime_s", lambda v: 1000 * v.mean()),
        enrichment=("enrichment", "mean"),
    ).round(3).reset_index()
    summ.to_csv(OUT / "comparison_table.csv", index=False)
    print(summ.to_string(index=False))
    return df


def exp_depth(seeds, max_p=5):
    rows = []
    for s in seeds:
        q = build_qubo(make_microgrid(8, seed=s))
        exact = constrained_optimum(q).x
        for r in depth_sweep(q, max_reps=max_p):
            rows.append(dict(seed=s, p=r.reps, p_optimal=r.p_optimal, p_near=r.p_near_optimal, enrichment=r.enrichment,
                             final_obj=r.history[-1], evals=r.n_evals,
                             found_opt=bool(np.array_equal(r.repaired_x, exact)),
                             two_q_depth=r.circuit_stats["two_qubit_depth"]))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "depth_sweep.csv", index=False)
    agg = df.groupby("p").mean(numeric_only=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(agg.index, agg["final_obj"], "o-")
    ax[0].set(xlabel="QAOA layers p", ylabel="normalised ⟨H⟩ (0 = optimum)", title="Energy vs depth")
    ax[1].plot(agg.index, agg["enrichment"], "o-", label="QAOA (CVaR α=0.25)")
    ax[1].axhline(1.0, ls="--", c="grey", label="random guess")
    ax[1].set(xlabel="QAOA layers p", ylabel="P(opt) / P(random)", title="Enrichment of the optimal dispatch")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "depth_sweep.png", dpi=150)
    print(agg.round(4).to_string())
    return df


def exp_density(seeds, n_values=(8, 10), keeps=(0.2, 0.4, 0.6, 0.8, 1.0), reps=2):
    """Convergence vs problem-graph density.

    The full QUBO couples every pair of assets (density 1). We keep only the
    strongest `keep` share of couplings, train QAOA on that sparser problem, and
    measure: optimiser evaluations to converge, enrichment of the TRUE optimum,
    whether the sparse model's own optimum is still the true one, and the
    transpiled two-qubit depth.
    """
    from gridq import problem_graph_density, sparsify_qubo
    from gridq.qaoa import _bind, build_ansatz, fast_qaoa_probs, ising_diagonal, optimise_parameters
    from gridq.qubo import energy_landscape
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    pm = generate_preset_pass_manager(optimization_level=3, basis_gates=["rz", "sx", "x", "cx"], seed_transpiler=0)
    rows = []
    for n in n_values:
        for s in seeds:
            q = build_qubo(make_microgrid(n, seed=s))
            full = energy_landscape(q)
            gs = np.isclose(full, full.min())
            for keep in keeps:
                qs = sparsify_qubo(q, keep)
                params, hist, nev = optimise_parameters(qs, reps, alpha=0.25, maxiter=1000)
                probs = fast_qaoa_probs(ising_diagonal(qs), n, params)
                ansatz, _ = build_ansatz(qs, reps)
                circ = _bind(ansatz, params)
                circ.measure_all()
                isa = pm.run(circ)
                rows.append(dict(
                    n=n, seed=s, keep=keep, density=round(problem_graph_density(qs), 3),
                    couplings=int(np.count_nonzero(qs.Q[np.triu_indices(n, 1)])),
                    evals_to_converge=nev, enrichment=float(probs[gs].sum() * gs.size / gs.sum()),
                    sparse_opt_is_true_opt=bool(np.argmin(energy_landscape(qs)) == np.argmin(full)),
                    two_q_gates=int(sum(v for k, v in isa.count_ops().items() if k in ("cx", "cz", "ecr"))),
                    two_q_depth=isa.depth(lambda i: i.operation.num_qubits == 2),
                ))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "density_sweep.csv", index=False)
    agg = df.groupby(["n", "keep"]).agg(
        density=("density", "mean"), couplings=("couplings", "mean"), evals=("evals_to_converge", "mean"),
        enrichment=("enrichment", "median"), model_exact=("sparse_opt_is_true_opt", "mean"),
        two_q_depth=("two_q_depth", "mean"),
    ).reset_index()
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    for n, sub in agg.groupby("n"):
        ax[0].plot(sub.density, sub.evals, "o-", label=f"n={n}")
        ax[1].plot(sub.density, sub.two_q_depth, "o-", label=f"n={n}")
        ax[2].plot(sub.density, sub.enrichment, "o-", label=f"n={n}")
        ax[3].plot(sub.density, 100 * sub.model_exact, "o-", label=f"n={n}")
    ax[0].set(xlabel="problem-graph density", ylabel="optimiser evaluations", title="Convergence vs density")
    ax[1].set(xlabel="problem-graph density", ylabel="2-qubit depth", title="Circuit cost vs density")
    ax[2].set(xlabel="problem-graph density", ylabel="median enrichment of true optimum", title="Quality vs density")
    ax[2].axhline(1, ls="--", c="grey")
    ax[3].set(xlabel="problem-graph density", ylabel="% instances", title="Sparse model still has the true optimum")
    for a in ax:
        a.legend()
    fig.tight_layout()
    fig.savefig(OUT / "density_sweep.png", dpi=150)
    print(agg.round(3).to_string(index=False))
    return df


def exp_scaling(seeds, n_values=(6, 8, 10, 12)):
    rows = []
    for n in n_values:
        for s in seeds:
            rows += solve_all(build_qubo(make_microgrid(n, seed=s)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "scaling.csv", index=False)
    agg = df.groupby(["solver", "n"]).agg(quality=("quality", "mean"), runtime_s=("runtime_s", "mean")).reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for name, sub in agg.groupby("solver"):
        ax[0].plot(sub.n, sub.quality, "o-", label=name)
        ax[1].semilogy(sub.n, sub.runtime_s, "o-", label=name)
    ax[0].set(xlabel="nodes (qubits)", ylabel="solution quality (1 = optimal)", title="Quality vs size")
    ax[1].set(xlabel="nodes (qubits)", ylabel="runtime (s)", title="Runtime vs size")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "scaling.png", dpi=150)
    print(agg.round(4).to_string(index=False))
    return df


def exp_noise(seeds, max_p=3):
    from gridq.hardware import backend_pass_manager, fake_backend, noisy_sampler

    backend = fake_backend()
    pm = backend_pass_manager(backend)
    rows = []
    for s in seeds:
        q = build_qubo(make_microgrid(6, seed=s))
        for p in range(1, max_p + 1):
            for label, sampler, pmgr in (("ideal", None, None), (f"noisy:{backend.name}", noisy_sampler(backend, seed=s), pm)):
                r = solve_qaoa(q, reps=p, sampler=sampler, pass_manager=pmgr, backend_name=label)
                rows.append(dict(seed=s, p=p, backend=label, p_optimal=r.p_optimal, enrichment=r.enrichment,
                                 expected_energy=r.expected_energy, two_q_gates=r.circuit_stats["two_qubit_gates"],
                                 two_q_depth=r.circuit_stats["two_qubit_depth"]))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "noise.csv", index=False)
    agg = df.groupby(["backend", "p"]).mean(numeric_only=True).reset_index()
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for b, sub in agg.groupby("backend"):
        ax.plot(sub.p, sub.enrichment, "o-", label=b)
    ax.axhline(1.0, ls="--", c="grey", label="random guess")
    ax.set(xlabel="QAOA layers p", ylabel="P(opt) / P(random)", title="Ideal vs noisy (n = 6)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "noise.png", dpi=150)
    print(agg.round(4).to_string(index=False))
    return df


def exp_day(village="village_n8_s0"):
    """Hour-by-hour dispatch of one village for a day, from its master-data files.

    Deficit per hour = sum of site demand - solar - wind (data/villages/<village>/).
    """
    from gridq.data import load_village

    v = load_village(Path(__file__).resolve().parents[1] / "data" / "villages" / village)
    bal = v.hourly_balance().set_index("hour")
    rows = []
    for h, r in bal.iterrows():
        if not (r.dispatch_needed and r.coverable):
            rows.append(dict(hour=h, deficit_kw=max(float(r.shortfall_kw), 0.0), solver="none_needed",
                             cost_inr=0.0, naive_cost_inr=0.0, demand_met_pct=100.0))
            continue
        g = v.microgrid(int(h))
        q = build_qubo(g)
        for name, x in (("qaoa_p3", solve_qaoa(q, reps=3, shots=256).repaired_x), ("greedy_operator", greedy_operator(q).x),
                        ("exact", constrained_optimum(q).x)):
            x = repair(g, x)
            m = dispatch_metrics(g, x)
            rows.append(dict(hour=h, deficit_kw=g.deficit, solver=name, cost_inr=m["cost_inr"],
                             naive_cost_inr=naive_policy_cost(g), demand_met_pct=m["demand_met_pct"]))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "deployment_day.csv", index=False)
    tot = df[df.solver != "none_needed"].groupby("solver")[["cost_inr", "naive_cost_inr"]].sum()
    tot["saved_pct"] = (100 * (1 - tot.cost_inr / tot.naive_cost_inr)).round(1)
    tot.to_csv(OUT / "deployment_day_summary.csv")
    print(f"village: {village}")
    print(tot.to_string())
    return df


EXPERIMENTS = {
    "correctness": exp_correctness, "depth": exp_depth, "density": exp_density,
    "scaling": exp_scaling, "noise": exp_noise, "day": exp_day,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=list(EXPERIMENTS), nargs="*")
    args = ap.parse_args()
    seeds = range(3) if args.quick else range(10)
    meta = {}
    for name in args.only or EXPERIMENTS:
        print(f"\n=== {name} ===")
        t0 = time.perf_counter()
        EXPERIMENTS[name]() if name == "day" else EXPERIMENTS[name](seeds)
        meta[name] = round(time.perf_counter() - t0, 1)
        print(f"[{name}] {meta[name]} s")
    (OUT / "run_meta.json").write_text(json.dumps({"seconds": meta, "quick": args.quick}, indent=2))


if __name__ == "__main__":
    main()
