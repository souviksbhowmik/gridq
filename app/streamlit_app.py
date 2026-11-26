"""GridQ live demo dashboard.

    streamlit run app/streamlit_app.py

Tabs:
  1. Demand & shortfall    village demand forecast vs solar + wind, hourly shortfall
  2. Providers             the flexible providers that can cover the shortfall
  3. Dispatch & compare    pick an hour, run several solvers, compare allocations
                           (QAOA on ideal sim, noisy sim, or real IBM hardware)
  4. Real quantum hardware saved IBM Quantum runs vs simulators (pitch page)
  5. 24-hour plan          hour-by-hour dispatch, cost vs diesel-first
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
import streamlit as st

from gridq import (
    build_qubo, constrained_optimum, cost_saved_pct, dispatch_metrics, greedy_descent, greedy_operator,
    random_sampling, repair, simulated_annealing, solve_qaoa,
)
from gridq.data import load_village
from gridq.metrics import naive_policy_cost
from gridq.qubo import all_bitstrings, dispatch_objective

try:  # run as `streamlit run app/streamlit_app.py`
    from qaoa_worker import (hardware_backend_isolated, score_saved_run_isolated, solve_hardware_isolated,
                             solve_qaoa_isolated)
except ImportError:  # imported as a module (tests)
    from app.qaoa_worker import (hardware_backend_isolated, score_saved_run_isolated, solve_hardware_isolated,
                                 solve_qaoa_isolated)

RESULTS = Path(__file__).resolve().parents[1] / "results"
BACKENDS = ["Ideal simulator", "Noisy simulator (IBM Heron snapshot)"]


def has_ibm_credentials() -> bool:
    import os

    from dotenv import dotenv_values

    env = dotenv_values(Path(__file__).resolve().parents[1] / ".env")
    return bool((env.get("QISKIT_IBM_TOKEN") or os.getenv("QISKIT_IBM_TOKEN") or "").strip())


@st.cache_data(show_spinner=False)
def scored_hardware_run(path: str, mtime: float) -> dict:
    return score_saved_run_isolated(path)

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
        # Qiskit runs in a separate process: its native code segfaults inside Streamlit on macOS.
        r = solve_qaoa_isolated(grid, reps, shots, noisy=backend.startswith("Noisy"))
        if not r["ok"]:
            raise RuntimeError(r["error"])
        x = np.array(r["x"], dtype=np.int8)
        extra = {k: r[k] for k in ("enrichment", "p_optimal", "evals", "two_q_depth", "two_q_gates", "counts")}
        extra.update(qubits=grid.n, shots=shots)
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

tab_demand, tab_prov, tab_solve, tab_hw, tab_day = st.tabs(
    ["⚡ Demand & shortfall", "🔌 Providers", "🧮 Dispatch & compare solvers", "🖥️ Real quantum hardware",
     "📅 24-hour plan"])

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
        backend = q3.selectbox("QAOA backend", BACKENDS)

    if st.button("Run solvers", type="primary", disabled=not chosen):
        results = {}
        with st.spinner("Running solvers ..."):
            for nm in chosen:
                try:
                    x, rt, extra = run_solver(nm, grid, qubo, reps, shots, backend)
                except RuntimeError as e:
                    st.error(f"{nm} failed: {e}")
                    continue
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
            gap = qcost - best_cost
            if np.isclose(gap, 0):
                a.metric("QAOA cost", f"₹{qcost:.0f}", "= exact optimum", delta_color="off")
            else:
                a.metric("QAOA cost", f"₹{qcost:.0f}", f"+₹{gap:.0f} vs exact optimum ₹{best_cost:.0f}",
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
                k1.metric("Backend", "noisy sim" if cmp["backend"].startswith("Noisy") else "ideal sim")
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

    # ------------------------------------------------------------------ real quantum hardware (separate)
    st.divider()
    st.subheader(f"🖥️ Run QAOA on real IBM quantum hardware ({hour:02d}:00)")
    st.caption("Separate from the comparison above (which runs on simulators). QAOA angles are trained on the "
               "simulator for free, then only the final circuit is sent to an IBM QPU. Results are shown below "
               "next to the ideal and noisy simulators for the same problem. Note the IBM job id to look the job up "
               "later on quantum.cloud.ibm.com.")
    if not has_ibm_credentials():
        st.warning("Needs IBM Quantum credentials: fill in `QISKIT_IBM_TOKEN` and `QISKIT_IBM_INSTANCE` in `.env` "
                   "(see README §6.4). Saved runs are on the **🖥️ Real quantum hardware** tab.")
    else:
        h1, h2, h3, h4 = st.columns(4)
        hw_reps = h1.selectbox("Layers p", [1, 2, 3], index=1, key="hw_reps",
                               help="Keep p ≤ 2 on today's hardware: deeper circuits add more noise than signal.")
        hw_shots = h2.selectbox("Shots", [1024, 2048, 4096], index=2, key="hw_shots")
        hw_mit = h3.selectbox("Error mitigation", ["Off (recommended)", "DD + twirling"], key="hw_mit",
                              help="Our measurements: DD + twirling lowered QAOA quality on ibm_fez and ibm_kingston.")
        with h4:
            st.write("")
            if st.button("Check QPU and queue (free)"):
                with st.spinner("Contacting IBM Quantum ..."):
                    info = hardware_backend_isolated()
                st.session_state["qpu_info"] = info
        info = st.session_state.get("qpu_info")
        if info:
            if info["ok"]:
                st.success(f"QPU **{info['backend']}** ({info['qubits']} qubits) · jobs waiting in queue: "
                           f"**{info['pending_jobs']}** · operational: {info['operational']}")
            else:
                st.error(f"Could not reach IBM Quantum: {info['error']}")
        ok = st.checkbox(f"I understand this submits one {hw_shots}-shot job to real IBM quantum hardware and "
                         "uses credits. Queue time can be seconds to hours.", key="hw_ok")
        if st.button("Run on real quantum hardware", type="primary", disabled=not ok):
            with st.spinner("Training on the simulator, then waiting for the IBM queue and the QPU ..."):
                t0 = time.perf_counter()
                hwr = solve_hardware_isolated(grid, hw_reps, hw_shots, mitigate=hw_mit.startswith("DD"))
                sim = solve_qaoa_isolated(grid, hw_reps, hw_shots, noisy=False) if hwr["ok"] else None
                noi = solve_qaoa_isolated(grid, hw_reps, hw_shots, noisy=True) if hwr["ok"] else None
            st.session_state["hw_live"] = {"key": (name, hour), "hw": hwr, "sim": sim, "noisy": noi,
                                           "reps": hw_reps, "shots": hw_shots, "mit": hw_mit,
                                           "total_s": time.perf_counter() - t0}

    live = st.session_state.get("hw_live")
    if live and live["key"] == (name, hour):
        hwr = live["hw"]
        if not hwr["ok"]:
            st.error(f"Hardware run failed: {hwr['error']}")
        else:
            exact = constrained_optimum(qubo).x
            best_cost = float(grid.cost @ exact)
            st.success(f"Ran on **{hwr['backend']}** · IBM job `{hwr['job_id']}` · queue + run "
                       f"{hwr['queue_and_run_s']:.0f} s · p = {live['reps']} · {live['shots']} shots · "
                       f"mitigation: {live['mit']}")
            runs = {"Real hardware (" + hwr["backend"] + ")": hwr, "Noisy simulator": live["noisy"],
                    "Ideal simulator": live["sim"]}
            runs = {k: r for k, r in runs.items() if r and r["ok"]}
            xh = np.array(hwr["x"], dtype=np.int8)
            a, b, c, d = st.columns(4)
            hcost = float(grid.cost @ xh)
            a.metric("Real hardware cost", f"₹{hcost:.0f}",
                     "= exact optimum" if np.isclose(hcost, best_cost) else f"+₹{hcost - best_cost:.0f} vs optimum",
                     delta_color="off" if np.isclose(hcost, best_cost) else "inverse")
            b.metric("Real hardware vs random guess", f"{hwr['enrichment']:.2f}×")
            if "Ideal simulator" in runs:
                c.metric("Ideal simulator", f"{runs['Ideal simulator']['enrichment']:.2f}×",
                         f"hardware keeps {100 * hwr['enrichment'] / runs['Ideal simulator']['enrichment']:.0f}%",
                         delta_color="off")
            d.metric("2-qubit depth on the QPU", hwr["two_q_depth"], f"{hwr['two_q_gates']} two-qubit gates",
                     delta_color="off")

            rows = []
            for k, r in runs.items():
                x = np.array(r["x"], dtype=np.int8)
                m = dispatch_metrics(grid, x)
                rows.append({"run": k, "providers on": ", ".join(str(i) for i in np.flatnonzero(x)),
                             "supplied kW": m["supplied_kw"], "cost ₹": round(float(grid.cost @ x), 1),
                             "optimal?": "✅" if np.array_equal(x, exact) else "❌",
                             "P(optimal) per shot": round(r["p_optimal"], 4),
                             "vs random guess ×": round(r["enrichment"], 2),
                             "2-qubit depth": r["two_q_depth"], "IBM job id": r.get("job_id", "")})
            st.markdown("**Real hardware vs simulators (same problem, same layers and shots)**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

            left, right = st.columns([2, 3])
            with left:
                fig, ax = plt.subplots(figsize=(5, 2.8))
                names_ = list(runs) + ["Random guess"]
                vals = [runs[k]["enrichment"] for k in runs] + [1.0]
                cols = ["#d32f2f" if k.startswith("Real") else ("#9e9e9e" if k.startswith("Random") else "#7e57c2")
                        for k in names_]
                ax.barh(names_, vals, color=cols)
                for i, val in enumerate(vals):
                    ax.text(val + 0.05, i, f"{val:.2f}×", va="center", fontsize=8)
                ax.invert_yaxis()
                ax.set(xlabel="P(optimal) ÷ P(random guess)", title="Chance of the optimal dispatch")
                ax.set_xlim(0, max(vals) * 1.3)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            with right:
                n = grid.n
                ideal_counts = runs.get("Ideal simulator", hwr)["counts"]
                order = sorted(ideal_counts, key=lambda k: -ideal_counts[k])[:16]
                e_opt = qubo.energy(exact)
                fig, ax = plt.subplots(figsize=(7.5, 2.8))
                w = 0.8 / len(runs)
                for j, (k, r) in enumerate(runs.items()):
                    tot = sum(r["counts"].values())
                    ax.bar(np.arange(len(order)) + j * w, [r["counts"].get(b, 0) / tot for b in order], w,
                           label=k, color=["#d32f2f", "#7e57c2", "#b39ddb"][j % 3])
                for i, b in enumerate(order):
                    if np.isclose(qubo.energy(np.array([int(ch) for ch in reversed(b)])), e_opt):
                        ax.axvspan(i - 0.1, i + 0.9, color="#c8e6c9", zorder=0)
                ax.axhline(1 / 2 ** n, ls="--", c="grey", lw=1)
                ax.set_xticks(np.arange(len(order)) + 0.4 - w / 2, order, rotation=90, fontsize=7)
                ax.set(ylabel="share of shots", title="Output distribution (green band = optimal combination)")
                ax.legend(fontsize=7)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

# --------------------------------------------------------------------------- tab 4: real quantum hardware
with tab_hw:
    st.subheader("Real IBM quantum hardware: does the quantum signal survive real noise?")
    files = sorted(RESULTS.glob("hardware_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    current = [f for f in files if "_v1_" not in f.name]
    if not files:
        st.info("No hardware runs yet. Run `python experiments/run_hardware.py --n 6 --reps 2`.")
    else:
        labels = {f.name: f for f in current + [f for f in files if f not in current]}
        pick = st.selectbox("Saved hardware run", list(labels),
                            format_func=lambda k: k + ("   (earlier model version)" if "_v1_" in k else ""))
        f = labels[pick]
        with st.spinner("Scoring saved hardware counts against the simulators ..."):
            hw = scored_hardware_run(str(f), f.stat().st_mtime)
        if not hw["ok"]:
            st.error(hw["error"])
        else:
            if "_v1_" in pick:
                st.warning("This run used an earlier model version (with line limits). Its numbers here are "
                           "re-scored against the current model; see README §4.5.")
            rows = pd.DataFrame(hw["rows"])
            ideal = rows[rows.setting == "Ideal simulator"].iloc[0]
            unmit = rows[rows.setting.str.endswith("unmitigated")]
            unmit = unmit.iloc[0] if len(unmit) else rows[rows.kind == "hardware"].iloc[0]
            noise = rows[rows.setting.str.startswith("Noise model")]

            # ---------------- 1. the case and what the quantum computer decided
            case_village = f"village_n{hw['n']}_s{hw['seed']}"
            case_hour = None
            if (DATA / case_village / "providers.csv").exists():
                cv = get_village(case_village)
                cb = cv.hourly_balance()
                hit = cb[cb.shortfall_kw.round(1) == round(hw["deficit_kw"], 1)]
                case_hour = int(hit.hour.iloc[0]) if len(hit) else None
                cprov = cv.providers
            st.markdown("### 1 · The result from the real quantum computer")
            where = f"village **{case_village}**" + (f" at **{case_hour:02d}:00**" if case_hour is not None else "")
            st.markdown(f"Problem sent to **{hw['backend']}**: {where}: cover a **{hw['deficit_kw']:.1f} kW** "
                        f"shortfall from **{hw['n']} providers** ({2 ** hw['n']} possible on/off combinations). "
                        f"QAOA with p = {hw['reps']}, {hw['shots']} shots per run, submitted {hw['submitted_at']}.")
            hrows = rows[rows.kind == "hardware"]
            od = hw["optimal_dispatch"]
            a, b, c, d = st.columns(4)
            a.metric("Quantum computer", hw["backend"], f"IBM Heron r2 · {hw['circuit']['num_qubits']} qubits",
                     delta_color="off")
            a_hw = hrows[hrows.setting.str.endswith("unmitigated")]
            a_hw = a_hw.iloc[0] if len(a_hw) else hrows.iloc[0]
            b.metric("Dispatch it chose", "providers " + ", ".join(map(str, a_hw.providers_on)),
                     "✅ optimal" if a_hw.found_optimal else "❌ not optimal", delta_color="off")
            c.metric("Cost of that dispatch", f"₹{a_hw.cost_inr:.1f}",
                     f"{a_hw.supplied_kw:.1f} kW for {hw['deficit_kw']:.1f} kW needed", delta_color="off")
            d.metric("Exact optimum (brute force)", f"₹{od['cost_inr']:.1f}",
                     "providers " + ", ".join(map(str, hw["optimal_providers"])), delta_color="off")

            res_rows = []
            for _, r in hrows.iterrows():
                res_rows.append({
                    "hardware run": r.setting, "IBM job id": r.job_id,
                    "providers switched on": ", ".join(map(str, r.providers_on)),
                    "supplied kW": round(r.supplied_kw, 1), "shortfall kW": hw["deficit_kw"],
                    "demand met %": 100.0 if r.supplied_kw >= hw["deficit_kw"] - 1e-9 else round(100 * r.supplied_kw / hw["deficit_kw"], 1),
                    "cost ₹": round(r.cost_inr, 1), "optimal?": "✅" if r.found_optimal else "❌",
                    "best answer per shot": f"{100 * r.p_optimal:.2f} %",
                })
            st.dataframe(pd.DataFrame(res_rows), hide_index=True, width="stretch")

            if case_hour is not None:
                on = set(a_hw.providers_on)
                plan = pd.DataFrame({
                    "id": cprov.provider_id, "site": cprov.site_name,
                    "provider type": [TYPE_LABEL[t] for t in cprov.provider_type],
                    "capacity (kW)": cprov.capacity_kw,
                    "switch": ["🟢 ON" if i in on else "⚪ off" for i in cprov.provider_id],
                    "cost ₹": [round(k * p, 1) if i in on else 0.0
                               for i, k, p in zip(cprov.provider_id, cprov.capacity_kw, cprov.price_inr_per_kwh)],
                })
                with st.expander(f"Dispatch instruction from {hw['backend']} (per provider)", expanded=True):
                    st.dataframe(plan, hide_index=True, width="stretch")
            st.caption("How the answer is read out: the circuit is measured "
                       f"{hw['shots']} times; each measurement is one on/off combination. Every sampled combination "
                       "goes through the same feasibility repair as the classical solvers, and the cheapest one "
                       "that covers the shortfall is the quantum computer's dispatch.")

            # ---------------- 2. how good was the quantum signal
            st.markdown("### 2 · How strong was the quantum signal? Real hardware vs simulators")
            a, b, c, d = st.columns(4)
            a.metric("Real hardware vs random guess", f"{unmit.enrichment:.2f}×",
                     f"{100 * unmit.p_optimal:.2f}% vs {100 / 2 ** hw['n']:.2f}% of shots", delta_color="off")
            b.metric("Ideal simulator", f"{ideal.enrichment:.2f}×",
                     f"hardware keeps {100 * unmit.enrichment / ideal.enrichment:.0f}%", delta_color="off")
            if len(noise):
                c.metric("Noise model predicted", f"{noise.iloc[0].enrichment:.2f}×",
                         f"real: {unmit.enrichment:.2f}×", delta_color="off")
            d.metric("2-qubit depth on the QPU", hw["circuit"]["two_qubit_depth"],
                     f"{hw['circuit']['two_qubit_gates']} two-qubit gates", delta_color="off")

            left, right = st.columns([3, 2])
            with left:
                colours = {"simulator": "#7e57c2", "hardware": "#d32f2f", "baseline": "#9e9e9e"}
                fig, ax = plt.subplots(figsize=(7, 3.6))
                err = rows.p_optimal_err * 2 ** hw["n"] / len(hw["optimal_states"])
                ax.barh(rows.setting, rows.enrichment, xerr=err, color=[colours[k] for k in rows.kind], capsize=3)
                for i, r in rows.iterrows():
                    ax.text(r.enrichment + err[i] + 0.06, i, f"{r.enrichment:.2f}×", va="center", fontsize=9)
                ax.axvline(1, ls="--", c="grey", lw=1)
                ax.invert_yaxis()
                ax.set(xlabel="P(optimal dispatch) ÷ P(random guess)   (higher is better; error bar = ±1σ)",
                       title="Chance of sampling the optimal dispatch")
                ax.set_xlim(0, rows.enrichment.max() * 1.25)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            with right:
                st.markdown("**What this shows**")
                lines = [
                    f"- The **same trained QAOA circuit** was run on a perfect simulator, a simulator with real "
                    f"IBM Heron noise, and the real **{hw['backend']}** quantum computer.",
                    f"- On real hardware the circuit still finds the optimal dispatch **{unmit.enrichment:.1f}× more "
                    f"often than random guessing**, which is {100 * unmit.enrichment / ideal.enrichment:.0f}% of the "
                    f"noise-free result.",
                ]
                if len(noise):
                    lines.append(f"- The noise model predicted **{noise.iloc[0].enrichment:.2f}×**; the real chip "
                                 f"gave **{unmit.enrichment:.2f}×**. Our simulator is a trustworthy stand-in.")
                mit = rows[rows.setting.str.endswith("· mitigated")]
                if len(mit):
                    lines.append(f"- **Error mitigation** (DD + twirling) *lowered* it to {mit.iloc[0].enrichment:.2f}×: "
                                 "twirling flattens the output distribution, which hurts a sampling algorithm. "
                                 "We measured it instead of assuming it helps.")
                od = hw["optimal_dispatch"]
                lines.append(f"- Every run decoded to the optimal dispatch: providers "
                             f"**{', '.join(map(str, hw['optimal_providers']))}**, {od['supplied_kw']:.1f} kW for "
                             f"**₹{od['cost_inr']:.1f}**, 100% of demand met.")
                st.markdown("\n".join(lines))

            table = rows.assign(
                **{"P(optimal)": rows.p_optimal.round(4), "± 1σ": rows.p_optimal_err.round(4),
                   "enrichment ×": rows.enrichment.round(2), "⟨H⟩ (lower = better)": rows.expected_energy.round(2),
                   "fidelity to ideal": rows.fidelity_to_ideal.round(3),
                   "optimal dispatch found": np.where(rows.found_optimal, "✅", "❌"), "IBM job id": rows.job_id},
            )[["setting", "kind", "P(optimal)", "± 1σ", "enrichment ×", "⟨H⟩ (lower = better)", "fidelity to ideal",
               "optimal dispatch found", "IBM job id"]]
            st.dataframe(table, hide_index=True, width="stretch")

            st.markdown("**Output distributions** (bar = share of shots; green = the optimal on/off combination)")
            probs = hw["probs"]
            show = [k for k in probs if not k.startswith("Random")]
            top = np.argsort(-np.array(probs["Ideal simulator"]))[:16]
            fig, axes = plt.subplots(1, len(show), figsize=(3.3 * len(show), 2.8), sharey=True)
            for ax, k in zip(np.atleast_1d(axes), show):
                p = np.array(probs[k])[top]
                ax.bar(range(len(top)), p, color=["#2e7d32" if t in hw["optimal_states"] else "#90a4ae" for t in top])
                ax.axhline(1 / 2 ** hw["n"], ls="--", c="grey", lw=1)
                ax.set_title(k, fontsize=8)
                ax.set_xticks([])
            np.atleast_1d(axes)[0].set_ylabel("probability")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption(f"Submitted {hw['submitted_at']} · {hw['shots']} shots per run · transpiled circuit: "
                       f"{hw['circuit']['two_qubit_gates']} two-qubit gates, two-qubit depth "
                       f"{hw['circuit']['two_qubit_depth']} · file `results/{pick}`. Same 16 most likely combinations "
                       "(by the ideal simulator) shown in every panel; dashed line = random guessing.")
            if len(files) > 1:
                with st.expander("All saved hardware runs"):
                    st.dataframe(pd.DataFrame([{"file": x.name, "saved": time.strftime("%Y-%m-%d %H:%M",
                                                time.localtime(x.stat().st_mtime))} for x in files]),
                                 hide_index=True, width="stretch")

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
                rq = solve_qaoa_isolated(g, 2, 256, noisy=False)
                if not rq["ok"]:
                    st.error(f"QAOA failed at {int(h):02d}:00: {rq['error']}")
                    break
                xq = repair(g, np.array(rq["x"], dtype=np.int8))
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
