"""GridQ live demo dashboard.

    streamlit run app/streamlit_app.py

Tabs:
  1. Demand & shortfall    village demand forecast vs solar + wind, hourly shortfall
  2. Providers             the flexible providers that can cover the shortfall
  3. Dispatch & compare    pick an hour, run several solvers, compare allocations
  4. 24-hour plan          hour-by-hour dispatch, cost vs diesel-first
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: the macOS GUI backend crashes in worker threads
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qiskit_aer  # noqa: F401  (import early: lazy import inside Streamlit's script thread segfaults on macOS)
import streamlit as st

from gridq import (
    build_qubo, constrained_optimum, cost_saved_pct, dispatch_metrics, greedy_descent, greedy_operator,
    random_sampling, repair, simulated_annealing, solve_qaoa,
)
from gridq.data import load_village
from gridq.metrics import naive_policy_cost
from gridq.qubo import all_bitstrings, dispatch_objective

warnings.filterwarnings("ignore")
st.set_page_config(page_title="GridQ: quantum microgrid dispatch", layout="wide")

DATA = Path(__file__).resolve().parents[1] / "data" / "villages"
TYPE_LABEL = {"battery": "🔋 battery", "diesel": "⛽ diesel genset", "demand_response": "🤝 demand response"}
TYPE_COLOUR = {"battery": "#43a047", "demand_response": "#1e88e5", "diesel": "#757575"}

SOLVERS = {
    "QAOA (quantum)": "Quantum: QAOA circuit trained with COBYLA, sampled on the chosen backend",
    "Exact (brute force)": "Classical: checks all 2ⁿ combinations, which is the ground truth",
    "Simulated annealing": "Classical metaheuristic: random bit flips with cooling",
    "Greedy (cheapest per kW)": "Classical rule of thumb: switch on the cheapest per kW until covered",
    "Greedy descent (QUBO)": "Classical local search on the same QUBO as QAOA",
    "Random sampling (control)": "Control: random on/off combinations with the same shot budget as QAOA",
}
QUANTUM = {"QAOA (quantum)"}


@st.cache_data
def village_names() -> list[str]:
    return sorted(p.name for p in DATA.iterdir() if (p / "providers.csv").exists())


@st.cache_resource
def get_village(name: str):
    return load_village(DATA / name)


def provider_labels(v) -> list[str]:
    return [f"{int(i)} · {n}" for i, n in zip(v.providers.provider_id, v.providers.site_name)]


def feasible_quality(grid, x) -> float:
    """1 = optimal dispatch, 0 = worst feasible dispatch."""
    X = all_bitstrings(grid.n)
    obj = dispatch_objective(grid, X)
    feas = X @ grid.power >= grid.deficit - 1e-9
    lo, hi = obj[feas].min(), obj[feas].max()
    return 1.0 if np.isclose(hi, lo) else float((hi - dispatch_objective(grid, x)) / (hi - lo))


def run_solver(name, grid, qubo, reps, shots, backend):
    t0 = time.perf_counter()
    extra = {}
    if name == "QAOA (quantum)":
        sampler = pm = None
        if backend.startswith("Noisy"):
            from gridq.hardware import backend_pass_manager, fake_backend, noisy_sampler

            fb = fake_backend()
            sampler, pm = noisy_sampler(fb, seed=0), backend_pass_manager(fb)
        r = solve_qaoa(qubo, reps=reps, shots=shots, sampler=sampler, pass_manager=pm)
        x = r.repaired_x
        extra = {"enrichment": r.enrichment, "p_optimal": r.p_optimal, "evals": r.n_evals,
                 "two_q_depth": r.circuit_stats["two_qubit_depth"], "two_q_gates": r.circuit_stats["two_qubit_gates"],
                 "qubits": grid.n, "shots": shots, "counts": r.counts}
    elif name == "Exact (brute force)":
        x = constrained_optimum(qubo).x
        extra = {"evals": 2 ** grid.n}
    elif name == "Simulated annealing":
        x = simulated_annealing(qubo, sweeps=500, restarts=5).x
        extra = {"evals": 500 * 5 * grid.n}
    elif name == "Greedy (cheapest per kW)":
        x = greedy_operator(qubo).x
    elif name == "Greedy descent (QUBO)":
        x = greedy_descent(qubo).x
    elif name == "Random sampling (control)":
        x = random_sampling(qubo, shots=shots, seed=0).x
        extra = {"evals": shots}
    runtime = time.perf_counter() - t0
    return repair(grid, x), runtime, extra


def draw_allocation(v, grid, results):
    """One row per solver: stacked kW of the providers it switched on, vs the shortfall."""
    order = np.argsort(grid.cost / grid.power)
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(results) + 1.2))
    names = list(results)
    for row, nm in enumerate(names):
        x = results[nm]["x"]
        left = 0.0
        for i in order:
            if x[i]:
                ax.barh(row, grid.power[i], left=left, color=TYPE_COLOUR[grid.assets[i]], edgecolor="white")
                ax.text(left + grid.power[i] / 2, row, str(i), ha="center", va="center", fontsize=8, color="white")
                left += grid.power[i]
        ax.text(left + 0.2, row, f"{left:.1f} kW · ₹{grid.cost @ x:.0f}", va="center", fontsize=8)
    ax.axvline(grid.deficit, color="#d32f2f", ls="--", lw=2)
    ax.text(grid.deficit, len(names) - 0.4, f" shortfall {grid.deficit:.1f} kW", color="#d32f2f", fontsize=8)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlabel("kW switched on (numbers = provider id; green battery · blue demand response · grey diesel)")
    ax.set_xlim(0, max(grid.deficit, max(results[n]["x"] @ grid.power for n in names)) * 1.35)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- sidebar
st.title("GridQ: QAOA load balancing for rural microgrids")
st.caption("QFF 2026 Quantum for Humanity · Climate track · Synthetic data in `data/villages/` "
           "(providers.csv, forecast.csv).")

names = village_names()
with st.sidebar:
    st.header("Village")
    name = st.selectbox("Village data", names,
                        index=names.index("village_n8_s0") if "village_n8_s0" in names else 0)
    v = get_village(name)
    bal = v.hourly_balance()
    need_hours = bal[bal.dispatch_needed & bal.coverable].hour.astype(int).tolist()
    peak_hour = int(bal.loc[bal.shortfall_kw.idxmax(), "hour"])
    hour = st.select_slider("Dispatch hour", options=need_hours, value=peak_hour,
                            format_func=lambda h: f"{h:02d}:00")
    st.caption("Only hours with a shortfall are listed.")

row = bal.set_index("hour").loc[hour]
grid = v.microgrid(hour)
qubo = build_qubo(grid)

tab_demand, tab_prov, tab_solve, tab_day = st.tabs(
    ["⚡ Demand & shortfall", "🔌 Providers", "🧮 Dispatch & compare solvers", "📅 24-hour plan"])

# --------------------------------------------------------------------------- tab 1: demand & shortfall
with tab_demand:
    st.subheader("Village demand forecast vs renewable supply")
    a, b, c, d = st.columns(4)
    a.metric("Peak demand", f"{bal.demand_kw.max():.1f} kW", f"at {int(bal.loc[bal.demand_kw.idxmax(), 'hour']):02d}:00",
             delta_color="off")
    b.metric("Peak shortfall", f"{bal.shortfall_kw.max():.1f} kW", f"at {peak_hour:02d}:00", delta_color="off")
    c.metric("Hours with a shortfall", f"{int(bal.dispatch_needed.sum())} / 24")
    d.metric("Shortfall energy per day", f"{bal.shortfall_kw.clip(lower=0).sum():.0f} kWh")

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.stackplot(bal.hour, bal.solar_kw, bal.wind_kw, labels=["solar", "wind"],
                 colors=["#ffd54f", "#81d4fa"], alpha=0.85)
    ax.plot(bal.hour, bal.demand_kw, "k-", lw=2.4, marker="o", ms=3, label="village demand (forecast)")
    ax.fill_between(bal.hour, bal.renewables_kw, bal.demand_kw, where=bal.shortfall_kw > 0,
                    color="#ef5350", alpha=0.3, label="shortfall → providers must cover")
    ax.axvline(hour, color="#6a1b9a", ls="--", lw=1.5)
    ax.text(hour + 0.15, ax.get_ylim()[1] * 0.92, f"selected {hour:02d}:00", color="#6a1b9a", fontsize=9)
    ax.set(xlabel="hour of day", ylabel="kW", xticks=range(0, 24, 2), xlim=(0, 23))
    ax.legend(loc="upper left", fontsize=8, ncol=4)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown(f"**At {hour:02d}:00:** demand **{row.demand_kw:.1f} kW** − solar **{row.solar_kw:.1f} kW** − "
                f"wind **{row.wind_kw:.1f} kW** = shortfall **{row.shortfall_kw:.1f} kW**")
    table = bal.rename(columns={
        "demand_kw": "demand (kW)", "solar_kw": "solar (kW)", "wind_kw": "wind (kW)",
        "renewables_kw": "solar + wind (kW)", "shortfall_kw": "shortfall (kW)", "dispatch_needed": "dispatch needed",
    }).drop(columns="coverable")
    table["shortfall (kW)"] = table["shortfall (kW)"].clip(lower=0)
    table.insert(0, "time", [f"{int(h):02d}:00" for h in table.hour])
    st.dataframe(table.drop(columns="hour"), hide_index=True, width="stretch", height=300)
    st.caption("Shortfall = demand − solar − wind (`forecast.csv`). Negative values mean renewables cover demand, "
               "so they are shown as 0. Every hour with a shortfall is one optimisation problem.")

# --------------------------------------------------------------------------- tab 2: providers
with tab_prov:
    st.subheader(f"Flexible providers: {v.n} that can be switched on to cover a shortfall")
    a, b, c, d = st.columns(4)
    a.metric("Providers (= qubits)", v.n)
    b.metric("Total capacity", f"{v.providers.capacity_kw.sum():.1f} kW")
    c.metric("Peak shortfall", f"{bal.shortfall_kw.max():.1f} kW")
    d.metric("Possible combinations", f"2^{v.n} = {2 ** v.n:,}")
    left, right = st.columns([3, 2])
    with left:
        show = pd.DataFrame({
            "id": v.providers.provider_id,
            "site": v.providers.site_name,
            "provider type": [TYPE_LABEL[t] for t in v.providers.provider_type],
            "capacity (kW)": v.providers.capacity_kw,
            "price (₹/kWh)": v.providers.price_inr_per_kwh.round(1),
            "cost if on 1 h (₹)": (v.providers.capacity_kw * v.providers.price_inr_per_kwh).round(1),
        })
        st.dataframe(show, hide_index=True, width="stretch")
        st.caption("`providers.csv`. Every provider feeds the shared village pool, so any of them can serve the "
                   "shortfall. Demand response adds power by cutting that site's own use.")
    with right:
        p = v.providers.sort_values("price_inr_per_kwh").reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(5.6, 4))
        ax.bar(range(len(p)), p.capacity_kw, color=[TYPE_COLOUR[t] for t in p.provider_type])
        for k, r in p.iterrows():
            ax.text(k, r.capacity_kw + 0.1, f"₹{r.price_inr_per_kwh:.0f}", ha="center", fontsize=7)
        ax.set_xticks(range(len(p)), p.provider_id.astype(int))
        ax.set(xlabel="provider id (cheapest per kWh first)", ylabel="capacity (kW)",
               title="Capacity and price (label = ₹/kWh)")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in TYPE_COLOUR.values()]
        ax.legend(handles, ["battery", "demand response", "diesel"], fontsize=7, loc="upper left")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# --------------------------------------------------------------------------- tab 3: dispatch & compare
with tab_solve:
    st.subheader(f"Dispatch for {hour:02d}:00: cover {grid.deficit:.1f} kW at least cost")
    st.markdown(f"Demand **{row.demand_kw:.1f} kW** − solar **{row.solar_kw:.1f}** − wind **{row.wind_kw:.1f}** "
                f"= shortfall **{grid.deficit:.1f} kW**, from **{v.n} providers** "
                f"(**{v.providers.capacity_kw.sum():.1f} kW** available), giving **{2 ** v.n:,}** possible on/off combinations.")

    c1, c2 = st.columns([3, 2])
    with c1:
        chosen = st.multiselect("Solvers to run", list(SOLVERS), default=list(SOLVERS),
                                help="\n".join(f"• {k}: {d}" for k, d in SOLVERS.items()))
    with c2:
        q1, q2, q3 = st.columns(3)
        reps = q1.selectbox("QAOA layers p", [1, 2, 3, 4, 5], index=2)
        shots = q2.selectbox("Shots", [64, 128, 256, 512, 1024, 4096], index=2)
        backend = q3.selectbox("QAOA backend", ["Ideal simulator", "Noisy simulator (IBM Heron snapshot)"])

    if st.button("Run solvers", type="primary", disabled=not chosen):
        results = {}
        with st.spinner("Running solvers ..."):
            for nm in chosen:
                x, rt, extra = run_solver(nm, grid, qubo, reps, shots, backend)
                results[nm] = {"x": x, "runtime": rt, **extra}
        st.session_state["cmp"] = {"key": (name, hour), "results": results, "backend": backend,
                                   "reps": reps, "shots": shots}

    cmp = st.session_state.get("cmp")
    if cmp and cmp["key"] == (name, hour):
        results = cmp["results"]
        exact = constrained_optimum(qubo).x
        best_cost = float(grid.cost @ exact)
        diesel = naive_policy_cost(grid)

        # ---- headline: quantum vs best classical
        if "QAOA (quantum)" in results:
            q = results["QAOA (quantum)"]
            qcost = float(grid.cost @ q["x"])
            classical = {k: r for k, r in results.items() if k not in QUANTUM and k != "Random sampling (control)"}
            a, b, c, d = st.columns(4)
            a.metric("QAOA cost", f"₹{qcost:.0f}", f"{qcost - best_cost:+.0f} vs exact optimum ₹{best_cost:.0f}",
                     delta_color="inverse")
            if classical:
                bc_name = min(classical, key=lambda k: float(grid.cost @ classical[k]["x"]))
                bc_cost = float(grid.cost @ classical[bc_name]["x"])
                b.metric("Best classical", f"₹{bc_cost:.0f}", bc_name, delta_color="off")
            c.metric("QAOA: P(optimal) vs random guess", f"{q['enrichment']:.1f}×")
            d.metric("QAOA circuit", f"{q['qubits']} qubits", f"2-qubit depth {q['two_q_depth']}", delta_color="off")

        # ---- allocation per solver
        st.markdown("**Allocation of providers by each solver**")
        st.pyplot(draw_allocation(v, grid, results))
        plt.close("all")

        alloc = pd.DataFrame({"provider": provider_labels(v),
                              "type": [TYPE_LABEL[t] for t in v.providers.provider_type],
                              "kW": v.providers.capacity_kw})
        for nm, r in results.items():
            alloc[nm] = np.where(r["x"] == 1, "🟢 ON", "·")
        with st.expander("Allocation table (which provider each solver switched on)", expanded=False):
            st.dataframe(alloc, hide_index=True, width="stretch")

        # ---- comparison table
        st.markdown("**Solver comparison**")
        rows = []
        for nm, r in results.items():
            x = r["x"]
            m = dispatch_metrics(grid, x)
            cost = float(grid.cost @ x)
            rows.append({
                "solver": nm,
                "kind": "quantum" if nm in QUANTUM else ("control" if "Random" in nm else "classical"),
                "providers on": ", ".join(str(i) for i in np.flatnonzero(x)),
                "supplied kW": m["supplied_kw"],
                "surplus kW": m["surplus_kw"],
                "demand met %": m["demand_met_pct"],
                "cost ₹": round(cost, 1),
                "gap to optimum ₹": round(cost - best_cost, 1),
                "optimal?": "✅" if np.array_equal(x, exact) else "❌",
                "quality (1 = optimal)": round(feasible_quality(grid, x), 3),
                "saved vs diesel-first %": cost_saved_pct(grid, x),
                "runtime ms": round(1000 * r["runtime"], 1),
                "evaluations": r.get("evals"),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"Diesel-first (status quo) cost this hour: ₹{diesel:.0f}. Every solver's output passes through "
                   "the same feasibility repair, so the comparison is like-for-like. **Quality** = 1 for the optimum "
                   "and 0 for the worst dispatch that still covers the shortfall.")

        # ---- quantum details
        if "QAOA (quantum)" in results:
            q = results["QAOA (quantum)"]
            with st.expander("Quantum details: QAOA output distribution and circuit", expanded=False):
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Backend", "noisy" if cmp["backend"].startswith("Noisy") else "ideal")
                k2.metric("Layers p / shots", f"{cmp['reps']} / {cmp['shots']}")
                k3.metric("P(optimal) per shot", f"{q['p_optimal']:.3f}", f"random: {1 / 2 ** grid.n:.4f}",
                          delta_color="off")
                k4.metric("2-qubit gates", q["two_q_gates"])
                top = sorted(q["counts"].items(), key=lambda kv: -kv[1])[:15]
                e_opt = qubo.energy(exact)
                e = [qubo.energy(np.array([int(ch) for ch in reversed(k)])) for k, _ in top]
                fig, ax = plt.subplots(figsize=(11, 2.8))
                ax.bar(range(len(top)), [val for _, val in top],
                       color=["#2e7d32" if np.isclose(ei, e_opt) else "#78909c" for ei in e])
                ax.set_xticks(range(len(top)), [k for k, _ in top], rotation=90, fontsize=7)
                ax.set(title="Most-sampled on/off combinations (green = optimal; bit i from the right = provider i)",
                       ylabel="shots")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        st.info("Honest take: at village scale (≤12 providers) exact search is instant and always optimal. QAOA finds "
                "the optimum and concentrates probability on it, but it is slower and its edge over random sampling "
                "is small at this size. Any advantage would need much larger problems than today's hardware runs reliably.")
    else:
        st.info("Choose solvers and press **Run solvers**.")

# --------------------------------------------------------------------------- tab 4: 24-hour plan
with tab_day:
    st.subheader("24-hour dispatch plan")
    st.caption("One optimisation per hour with a shortfall. QAOA runs on the ideal simulator (p = 2, 256 shots) "
               "to keep this fast (about 5 s).")
    if st.button("Plan the whole day"):
        out = []
        prog = st.progress(0.0)
        b2 = bal.set_index("hour")
        for k, h in enumerate(b2.index):
            r = b2.loc[h]
            if not (r.dispatch_needed and r.coverable):
                out.append({"hour": int(h), "shortfall kW": 0.0, "providers on (QAOA)": "none needed",
                            "QAOA ₹": 0.0, "exact ₹": 0.0, "greedy ₹": 0.0, "diesel-first ₹": 0.0,
                            "QAOA optimal": True})
            else:
                g = v.microgrid(int(h))
                qb = build_qubo(g)
                xq = repair(g, solve_qaoa(qb, reps=2, shots=256).repaired_x)
                xe = repair(g, constrained_optimum(qb).x)
                xo = repair(g, greedy_operator(qb).x)
                out.append({"hour": int(h), "shortfall kW": g.deficit,
                            "providers on (QAOA)": ", ".join(str(i) for i in np.flatnonzero(xq)),
                            "QAOA ₹": float(g.cost @ xq), "exact ₹": float(g.cost @ xe),
                            "greedy ₹": float(g.cost @ xo), "diesel-first ₹": naive_policy_cost(g),
                            "QAOA optimal": bool(np.array_equal(xq, xe))})
            prog.progress((k + 1) / len(b2))
        st.session_state["day"] = (name, pd.DataFrame(out))

    day = st.session_state.get("day")
    if day and day[0] == name:
        df = day[1]
        tot = df[["QAOA ₹", "exact ₹", "greedy ₹", "diesel-first ₹"]].sum()
        a, b, c, d = st.columns(4)
        a.metric("Diesel-first (status quo)", f"₹{tot['diesel-first ₹']:,.0f}")
        b.metric("Greedy (cheapest per kW)", f"₹{tot['greedy ₹']:,.0f}",
                 f"-{100 * (1 - tot['greedy ₹'] / tot['diesel-first ₹']):.0f}%")
        c.metric("QAOA", f"₹{tot['QAOA ₹']:,.0f}", f"-{100 * (1 - tot['QAOA ₹'] / tot['diesel-first ₹']):.0f}%")
        d.metric("QAOA = exact optimum", f"{int(df['QAOA optimal'].sum())} / 24 h")
        fig, ax = plt.subplots(figsize=(11, 3.4))
        w = 0.28
        ax.bar(df.hour - w, df["diesel-first ₹"], w, label="diesel-first", color="#bdbdbd")
        ax.bar(df.hour, df["greedy ₹"], w, label="greedy", color="#90caf9")
        ax.bar(df.hour + w, df["QAOA ₹"], w, label="QAOA", color="#2e7d32")
        ax.set(xlabel="hour", ylabel="₹ per hour", xticks=range(0, 24, 2))
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(df, hide_index=True, width="stretch")
    else:
        st.info("Press **Plan the whole day** to dispatch every hour.")
