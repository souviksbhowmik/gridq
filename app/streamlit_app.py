"""GridQ live demo dashboard.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")  # headless backend: the macOS GUI backend crashes in worker threads
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import qiskit_aer  # noqa: F401  (import early: lazy import inside Streamlit's script thread segfaults on macOS)
import streamlit as st

from gridq import (
    build_qubo, constrained_optimum, cost_saved_pct, dispatch_metrics, greedy_operator,
    make_microgrid, random_sampling, repair, simulated_annealing, solve_qaoa,
)
from gridq.metrics import naive_policy_cost

warnings.filterwarnings("ignore")
st.set_page_config(page_title="GridQ: quantum microgrid dispatch", layout="wide")

st.title("GridQ: QAOA load balancing for rural microgrids")
st.caption("QFF 2026 Quantum for Humanity: Climate track. All data is synthetic.")

with st.sidebar:
    st.header("Village microgrid")
    n = st.slider("Nodes (= qubits)", 6, 12, 8)
    density = st.slider("Line density", 0.2, 1.0, 0.4, 0.05)
    seed = st.number_input("Scenario seed", 0, 999, 0)
    st.header("QAOA")
    reps = st.slider("Layers p", 1, 5, 3)
    shots = st.select_slider("Shots", [64, 128, 256, 512, 1024, 4096], 256)
    backend = st.radio("Backend", ["Ideal simulator", "Noisy simulator (IBM Heron snapshot)"])
    run = st.button("Solve dispatch", type="primary")

grid = make_microgrid(n, density, seed=int(seed))
qubo = build_qubo(grid)

ICON = {"household": "🏠", "school": "🏫", "water_pump": "🚰", "cold_store": "❄️", "clinic": "🏥"}
nodes = pd.DataFrame({
    "node": range(n),
    "site": [f"{ICON[k]} {k}" for k in grid.kinds],
    "asset": grid.assets,
    "power_kw": grid.power,
    "cost_inr": grid.cost,
    "critical": grid.priority.astype(bool),
})


def draw(x=None, title=""):
    fig, ax = plt.subplots(figsize=(5, 4))
    pos = nx.spring_layout(grid.graph, seed=1)
    colours = ["#2e7d32" if (x is not None and x[i]) else "#cfd8dc" for i in range(n)]
    over = [e for e, s in grid.line_stress.items() if x is not None and s > 0 and x[e[0]] and x[e[1]]]
    nx.draw_networkx_edges(grid.graph, pos, ax=ax, edge_color="#90a4ae")
    nx.draw_networkx_edges(grid.graph, pos, edgelist=over, ax=ax, edge_color="#d32f2f", width=3)
    nx.draw_networkx_nodes(grid.graph, pos, ax=ax, node_color=colours, node_size=650,
                           edgecolors=["#c62828" if c else "#455a64" for c in grid.priority])
    nx.draw_networkx_labels(grid.graph, pos, ax=ax,
                            labels={i: f"{i}\n{grid.power[i]:.0f}kW" for i in range(n)}, font_size=7)
    ax.set_title(title)
    ax.axis("off")
    return fig


c1, c2 = st.columns([1, 1])
with c1:
    st.subheader("Evening peak at 19:00")
    st.metric("Power deficit to cover", f"{grid.deficit:.1f} kW")
    st.dataframe(nodes, hide_index=True, width="stretch")
with c2:
    st.pyplot(draw(None, "Distribution grid (red border = critical load)"))

if run:
    with st.spinner("Training QAOA angles and sampling ..."):
        sampler = pm = None
        if backend.startswith("Noisy"):
            from gridq.hardware import backend_pass_manager, fake_backend, noisy_sampler

            fb = fake_backend()
            sampler, pm = noisy_sampler(fb, seed=int(seed)), backend_pass_manager(fb)
        q = solve_qaoa(qubo, reps=reps, shots=shots, sampler=sampler, pass_manager=pm)
        exact = constrained_optimum(qubo).x
        sols = {
            "QAOA (quantum)": q.repaired_x,
            "Exact optimum": exact,
            "Simulated annealing": simulated_annealing(qubo, sweeps=500, restarts=5).x,
            "Operator rule (cheapest first)": greedy_operator(qubo).x,
            "Random sampling (control)": random_sampling(qubo, shots=shots, seed=int(seed)).x,
        }
        rows = []
        for name, x in sols.items():
            x = repair(grid, x)
            m = dispatch_metrics(grid, x)
            rows.append({"solver": name, "optimal": bool(np.array_equal(x, exact)),
                         "demand met %": m["demand_met_pct"], "cost ₹": m["cost_inr"],
                         "saved vs diesel-first %": cost_saved_pct(grid, x),
                         "overloaded lines": m["overloaded_lines"], "critical sites on": m["critical_assets_on"]})

    st.divider()
    a, b, c, d = st.columns(4)
    qm = dispatch_metrics(grid, q.repaired_x)
    a.metric("Demand met", f"{qm['demand_met_pct']:.0f} %")
    b.metric("Cost vs diesel-first", f"₹{qm['cost_inr']:.0f}", f"-{cost_saved_pct(grid, q.repaired_x):.0f} %")
    c.metric("P(optimal) vs random guess", f"{q.enrichment:.1f}×")
    d.metric("2-qubit depth", q.circuit_stats["two_qubit_depth"])

    l, r = st.columns([1, 1])
    with l:
        st.pyplot(draw(q.repaired_x, "QAOA dispatch (green = asset on, red line = overload)"))
    with r:
        top = sorted(q.counts.items(), key=lambda kv: -kv[1])[:15]
        fig, ax = plt.subplots(figsize=(5, 4))
        e = [qubo.energy(np.array([int(c) for c in reversed(k)])) for k, _ in top]
        ax.bar(range(len(top)), [v for _, v in top],
               color=["#2e7d32" if np.isclose(ei, qubo.energy(exact)) else "#78909c" for ei in e])
        ax.set_xticks(range(len(top)), [k for k, _ in top], rotation=90, fontsize=7)
        ax.set(title="Most-sampled dispatches (green = optimal)", ylabel="shots")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Solver comparison")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Naive diesel-first policy cost: ₹{naive_policy_cost(grid):.0f}. Every solver's output goes "
               "through the same feasibility repair, so the comparison is like-for-like.")
    st.info("Honest take: at village scale (≤12 nodes) classical solvers are exact and instant. QAOA "
            "concentrates probability on good dispatches, but its advantage would have to come at "
            "scales today's hardware can't yet run reliably. See README → Limitations.")
