"""GridQ live demo dashboard.

    streamlit run app/streamlit_app.py

Tabs:
  1. Village master data   sites, flexible assets, lines (from data/villages/*.csv)
  2. Demand & supply       hourly demand per site, solar, wind, the deficit to dispatch
  3. Dispatch (one hour)   QAOA vs classical solvers for the selected hour
  4. 24-hour plan          hour-by-hour dispatch, cost vs diesel-first
"""

from __future__ import annotations

import warnings
from pathlib import Path

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
    random_sampling, repair, simulated_annealing, solve_qaoa,
)
from gridq.data import load_village
from gridq.metrics import naive_policy_cost

warnings.filterwarnings("ignore")
st.set_page_config(page_title="GridQ: quantum microgrid dispatch", layout="wide")

DATA = Path(__file__).resolve().parents[1] / "data" / "villages"
ICON = {"household": "🏠", "school": "🏫", "water_pump": "🚰", "cold_store": "❄️", "clinic": "🏥"}
ASSET_ICON = {"battery": "🔋 battery", "diesel": "⛽ diesel genset", "demand_response": "🤝 demand response"}
CRIT = "#c62828"


@st.cache_data
def village_names() -> list[str]:
    return sorted(p.name for p in DATA.iterdir() if (p / "sites.csv").exists())


@st.cache_resource
def get_village(name: str):
    return load_village(DATA / name)


def draw_grid(v, x=None, title="", figsize=(5.2, 4.2)):
    g = nx.Graph()
    g.add_nodes_from(range(v.n))
    over = []
    p = v.sites.asset_kw.to_numpy()
    for r in v.lines.itertuples():
        i, j = int(r.from_site), int(r.to_site)
        g.add_edge(i, j)
        if x is not None and x[i] and x[j] and p[i] + p[j] > r.capacity_kw:
            over.append((i, j))
    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(g, seed=1)
    colours = ["#e3f2fd"] * v.n if x is None else ["#2e7d32" if x[i] else "#eceff1" for i in range(v.n)]
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#90a4ae", width=1.5)
    nx.draw_networkx_edges(g, pos, edgelist=over, ax=ax, edge_color="#d32f2f", width=3.5)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colours, node_size=900,
                           edgecolors=[CRIT if c else "#455a64" for c in v.sites.critical], linewidths=2.2)
    font = ["white" if (x is not None and x[i]) else "black" for i in range(v.n)]
    for i in range(v.n):
        ax.text(*pos[i], f"{i}\n{p[i]:.1f}kW", ha="center", va="center", fontsize=7, color=font[i])
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    return fig


def site_labels(v) -> list[str]:
    return [f"{ICON[t]} {n}" for t, n in zip(v.sites.site_type, v.sites.site_name)]


# --------------------------------------------------------------------------- sidebar
st.title("GridQ: QAOA load balancing for rural microgrids")
st.caption("QFF 2026 Quantum for Humanity · Climate track · All data is synthetic, "
           "stored as editable CSVs in `data/villages/`.")

names = village_names()
with st.sidebar:
    st.header("1 · Village")
    name = st.selectbox("Village master data", names,
                        index=names.index("village_n8_s0") if "village_n8_s0" in names else 0)
    v = get_village(name)
    bal = v.hourly_balance()
    need_hours = bal[bal.dispatch_needed & bal.coverable].hour.astype(int).tolist()
    peak_hour = int(bal.loc[bal.net_deficit_kw.idxmax(), "hour"])
    hour = st.select_slider("Dispatch hour", options=need_hours, value=peak_hour,
                            format_func=lambda h: f"{h:02d}:00")
    st.caption("Only hours where renewables fall short are listed.")
    st.header("2 · QAOA")
    reps = st.slider("Layers p", 1, 5, 3)
    shots = st.select_slider("Shots", [64, 128, 256, 512, 1024, 4096], 256)
    backend = st.radio("Backend", ["Ideal simulator", "Noisy simulator (IBM Heron snapshot)"])

row = bal.set_index("hour").loc[hour]
grid = v.microgrid(hour)
qubo = build_qubo(grid)

tab_data, tab_demand, tab_solve, tab_day = st.tabs(
    ["📋 Village master data", "⚡ Demand & supply", "🧮 Dispatch (one hour)", "📅 24-hour plan"])

# --------------------------------------------------------------------------- tab 1: master data
with tab_data:
    st.subheader(f"{name}: {v.n} sites")
    a, b, c, d = st.columns(4)
    a.metric("Sites", v.n)
    b.metric("Critical sites", int(v.sites.critical.sum()))
    c.metric("Flexible capacity", f"{v.sites.asset_kw.sum():.1f} kW")
    d.metric("Solar PV installed", f"{v.meta.get('solar_capacity_kw', 0):.1f} kW")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Sites and their flexible assets** · `sites.csv`")
        show = pd.DataFrame({
            "id": v.sites.site_id,
            "site": site_labels(v),
            "critical": v.sites.critical,
            "peak demand (kW)": v.sites.peak_demand_kw,
            "flexible asset": [ASSET_ICON[a] for a in v.sites.asset],
            "asset output (kW)": v.sites.asset_kw,
            "price (₹/kWh)": v.sites.price_inr_per_kwh.round(1),
            "cost if on 1 h (₹)": (v.sites.asset_kw * v.sites.price_inr_per_kwh).round(1),
        })
        st.dataframe(show, hide_index=True, width="stretch")
        st.caption("**Peak demand** = the most this site consumes in a day. **Asset output** = the extra power "
                   "its flexible asset can feed into the shared village grid when switched on.")
        st.markdown("**Distribution lines** · `lines.csv`")
        pk = v.sites.asset_kw.to_numpy()
        ln = pd.DataFrame({
            "from": v.lines.from_site, "to": v.lines.to_site, "capacity (kW)": v.lines.capacity_kw,
            "both assets on (kW)": [round(pk[i] + pk[j], 1) for i, j in zip(v.lines.from_site, v.lines.to_site)],
        })
        ln["overloads if both on"] = ln["both assets on (kW)"] > ln["capacity (kW)"]
        st.dataframe(ln, hide_index=True, width="stretch")
    with right:
        st.pyplot(draw_grid(v, None, "Village grid (red ring = critical site, label = asset kW)"))
        plt.close("all")
        st.markdown("**Flexible assets by type**")
        summary = v.sites.groupby("asset").agg(
            sites=("site_id", "size"), total_kW=("asset_kw", "sum"), avg_price_per_kWh=("price_inr_per_kwh", "mean"),
        ).round(1)
        summary.index = [ASSET_ICON[i] for i in summary.index]
        st.dataframe(summary, width="stretch")
        st.caption(f"Files: `data/villages/{name}/`: sites.csv, lines.csv, site_demand.csv, renewables.csv")

# --------------------------------------------------------------------------- tab 2: demand
with tab_demand:
    st.subheader("Village demand vs renewable supply")
    a, b, c, d = st.columns(4)
    a.metric(f"Demand at {hour:02d}:00", f"{row.demand_kw:.1f} kW")
    b.metric("Solar + wind", f"{row.solar_kw + row.wind_kw:.1f} kW")
    c.metric("Deficit to dispatch", f"{row.net_deficit_kw:.1f} kW")
    d.metric("Hours needing dispatch", f"{int(bal.dispatch_needed.sum())} / 24")

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.stackplot(bal.hour, bal.solar_kw, bal.wind_kw, labels=["solar", "wind"],
                 colors=["#ffd54f", "#81d4fa"], alpha=0.85)
    ax.plot(bal.hour, bal.demand_kw, "k-", lw=2.2, label="village demand")
    ax.fill_between(bal.hour, bal.solar_kw + bal.wind_kw, bal.demand_kw,
                    where=bal.net_deficit_kw > 0, color="#ef5350", alpha=0.25, label="deficit → dispatch")
    ax.axvline(hour, color="#6a1b9a", ls="--", lw=1.5)
    ax.text(hour + 0.15, ax.get_ylim()[1] * 0.92, f"selected {hour:02d}:00", color="#6a1b9a", fontsize=9)
    ax.set(xlabel="hour of day", ylabel="kW", xticks=range(0, 24, 2), xlim=(0, 23))
    ax.legend(loc="upper left", fontsize=8, ncol=4)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    left, right = st.columns([3, 2])
    labels = site_labels(v)
    with left:
        st.markdown("**Hourly demand per site** (kW) · `site_demand.csv`")
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.stackplot(bal.hour, v.demand.T.to_numpy(), labels=labels, alpha=0.9)
        ax.axvline(hour, color="#6a1b9a", ls="--", lw=1.5)
        ax.set(xlabel="hour", ylabel="kW", xticks=range(0, 24, 2), xlim=(0, 23))
        ax.legend(fontsize=6.5, loc="upper left", ncol=2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown(f"**Who is consuming at {hour:02d}:00**")
        now = pd.DataFrame({"site": labels, "demand (kW)": v.demand.loc[hour].to_numpy(),
                            "critical": v.sites.critical.to_numpy()})
        now["share %"] = (100 * now["demand (kW)"] / now["demand (kW)"].sum()).round(1)
        st.dataframe(now.sort_values("demand (kW)", ascending=False), hide_index=True, width="stretch")

    with st.expander("Hourly balance table (all 24 hours)"):
        st.dataframe(bal, hide_index=True, width="stretch")
    st.caption("Deficit = Σ site demand − solar − wind. Each hour with a deficit becomes one optimisation: "
               "choose which flexible assets to switch on to cover it.")

# --------------------------------------------------------------------------- tab 3: one-hour dispatch
with tab_solve:
    st.subheader(f"Dispatch for {hour:02d}:00: cover {grid.deficit:.1f} kW at least cost")
    st.markdown(f"Demand **{row.demand_kw:.1f} kW** − solar **{row.solar_kw:.1f}** − wind **{row.wind_kw:.1f}** "
                f"= deficit **{grid.deficit:.1f} kW**, to be covered from **{v.n} flexible assets** "
                f"(**{v.sites.asset_kw.sum():.1f} kW** available): **2^{v.n} = {2**v.n}** possible on/off combinations.")
    if st.button("Solve dispatch", type="primary"):
        with st.spinner("Training QAOA angles and sampling ..."):
            sampler = pm = None
            if backend.startswith("Noisy"):
                from gridq.hardware import backend_pass_manager, fake_backend, noisy_sampler

                fb = fake_backend()
                sampler, pm = noisy_sampler(fb, seed=0), backend_pass_manager(fb)
            q = solve_qaoa(qubo, reps=reps, shots=shots, sampler=sampler, pass_manager=pm)
            exact = constrained_optimum(qubo).x
            sols = {
                "QAOA (quantum)": q.repaired_x,
                "Exact optimum": exact,
                "Simulated annealing": simulated_annealing(qubo, sweeps=500, restarts=5).x,
                "Operator rule (cheapest first)": greedy_operator(qubo).x,
                "Random sampling (control)": random_sampling(qubo, shots=shots, seed=0).x,
            }
            rows = []
            for nm, xs in sols.items():
                xs = repair(grid, xs)
                m = dispatch_metrics(grid, xs)
                rows.append({"solver": nm, "optimal": bool(np.array_equal(xs, exact)),
                             "assets on": ", ".join(str(i) for i in np.flatnonzero(xs)),
                             "supplied kW": m["supplied_kw"], "demand met %": m["demand_met_pct"],
                             "cost ₹": m["cost_inr"], "saved vs diesel-first %": cost_saved_pct(grid, xs),
                             "overloaded lines": m["overloaded_lines"], "critical sites on": m["critical_assets_on"]})
        st.session_state["result"] = (name, hour, backend, q, rows, exact)

    res = st.session_state.get("result")
    if res and res[0] == name and res[1] == hour:
        _, _, used_backend, q, rows, exact = res
        x = q.repaired_x
        qm = dispatch_metrics(grid, x)
        st.caption(f"Backend: {used_backend} · p = {q.reps}")
        a, b, c, d = st.columns(4)
        a.metric("Demand met", f"{qm['demand_met_pct']:.0f} %",
                 f"{qm['supplied_kw']:.1f} of {grid.deficit:.1f} kW", delta_color="off")
        b.metric("Cost this hour", f"₹{qm['cost_inr']:.0f}",
                 f"-{cost_saved_pct(grid, x):.0f}% vs diesel-first ₹{naive_policy_cost(grid):.0f}")
        c.metric("P(optimal) vs random guess", f"{q.enrichment:.1f}×")
        d.metric("2-qubit depth", q.circuit_stats["two_qubit_depth"])

        left, right = st.columns([1, 1])
        with left:
            st.pyplot(draw_grid(v, x, "QAOA dispatch (green = asset on, red line = overload)"))
            plt.close("all")
        with right:
            st.markdown("**Dispatch instruction for the operator**")
            plan = pd.DataFrame({
                "id": v.sites.site_id, "site": site_labels(v),
                "asset": [ASSET_ICON[a] for a in v.sites.asset], "kW": v.sites.asset_kw,
                "switch": np.where(x == 1, "🟢 ON", "⚪ off"), "cost ₹": np.where(x == 1, grid.cost, 0.0),
            })
            st.dataframe(plan, hide_index=True, width="stretch")
        top = sorted(q.counts.items(), key=lambda kv: -kv[1])[:15]
        e_opt = qubo.energy(exact)
        e = [qubo.energy(np.array([int(ch) for ch in reversed(k)])) for k, _ in top]
        fig, ax = plt.subplots(figsize=(11, 2.8))
        ax.bar(range(len(top)), [val for _, val in top],
               color=["#2e7d32" if np.isclose(ei, e_opt) else "#78909c" for ei in e])
        ax.set_xticks(range(len(top)), [k for k, _ in top], rotation=90, fontsize=7)
        ax.set(title="Most-sampled dispatches from the quantum circuit (green = optimal)", ylabel="shots")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("**Solver comparison** (every solver passes through the same feasibility repair)")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.info("Honest take: at village scale (≤12 assets) classical solvers are exact and instant. QAOA "
                "concentrates probability on good dispatches, but any advantage would have to come at "
                "scales today's hardware can't yet run reliably. See README → Limitations.")
    else:
        st.info("Pick an hour in the sidebar and press **Solve dispatch**.")

# --------------------------------------------------------------------------- tab 4: 24-hour plan
with tab_day:
    st.subheader("24-hour dispatch plan")
    st.caption("One optimisation per hour with a deficit. QAOA runs on the ideal simulator (p = 2, 256 shots) "
               "to keep this fast (about 5 s).")
    if st.button("Plan the whole day"):
        out = []
        prog = st.progress(0.0)
        b2 = bal.set_index("hour")
        for k, h in enumerate(b2.index):
            r = b2.loc[h]
            if not (r.dispatch_needed and r.coverable):
                out.append({"hour": int(h), "deficit kW": 0.0, "assets on (QAOA)": "none needed",
                            "QAOA ₹": 0.0, "exact ₹": 0.0, "operator rule ₹": 0.0, "diesel-first ₹": 0.0,
                            "QAOA optimal": True})
            else:
                g = v.microgrid(int(h))
                qb = build_qubo(g)
                xq = repair(g, solve_qaoa(qb, reps=2, shots=256).repaired_x)
                xe = repair(g, constrained_optimum(qb).x)
                xo = repair(g, greedy_operator(qb).x)
                out.append({"hour": int(h), "deficit kW": g.deficit,
                            "assets on (QAOA)": ", ".join(str(i) for i in np.flatnonzero(xq)),
                            "QAOA ₹": float(g.cost @ xq), "exact ₹": float(g.cost @ xe),
                            "operator rule ₹": float(g.cost @ xo), "diesel-first ₹": naive_policy_cost(g),
                            "QAOA optimal": bool(np.array_equal(xq, xe))})
            prog.progress((k + 1) / len(b2))
        st.session_state["day"] = (name, pd.DataFrame(out))

    day = st.session_state.get("day")
    if day and day[0] == name:
        df = day[1]
        tot = df[["QAOA ₹", "exact ₹", "operator rule ₹", "diesel-first ₹"]].sum()
        a, b, c, d = st.columns(4)
        a.metric("Diesel-first (status quo)", f"₹{tot['diesel-first ₹']:,.0f}")
        b.metric("Operator rule", f"₹{tot['operator rule ₹']:,.0f}",
                 f"-{100 * (1 - tot['operator rule ₹'] / tot['diesel-first ₹']):.0f}%")
        c.metric("QAOA", f"₹{tot['QAOA ₹']:,.0f}", f"-{100 * (1 - tot['QAOA ₹'] / tot['diesel-first ₹']):.0f}%")
        d.metric("QAOA = exact optimum", f"{int(df['QAOA optimal'].sum())} / 24 h")
        fig, ax = plt.subplots(figsize=(11, 3.4))
        w = 0.28
        ax.bar(df.hour - w, df["diesel-first ₹"], w, label="diesel-first", color="#bdbdbd")
        ax.bar(df.hour, df["operator rule ₹"], w, label="operator rule", color="#90caf9")
        ax.bar(df.hour + w, df["QAOA ₹"], w, label="QAOA", color="#2e7d32")
        ax.set(xlabel="hour", ylabel="₹ per hour", xticks=range(0, 24, 2))
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(df, hide_index=True, width="stretch")
    else:
        st.info("Press **Plan the whole day** to dispatch every hour.")
