"""Synthetic rural-microgrid instances (no real data is used anywhere).

One dispatch interval (e.g. the 19:00 evening peak) is modelled. Each node owns
one flexible asset that can be switched on (x_i = 1) to inject power:
battery discharge, a small diesel genset, or a demand-response contract.
The operator must cover the net deficit D = demand - renewable generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

# name, power range (kW), cost per kWh range (INR), share of nodes
ASSET_TYPES = {
    "battery": dict(power=(2.0, 5.0), cost=(4.0, 7.0)),
    "diesel": dict(power=(4.0, 8.0), cost=(22.0, 30.0)),
    "demand_response": dict(power=(1.0, 3.0), cost=(10.0, 14.0)),
}
NODE_KINDS = ["household", "household", "household", "school", "water_pump", "cold_store", "clinic"]
CRITICAL_KINDS = {"clinic", "water_pump", "cold_store"}


@dataclass
class Microgrid:
    """A single microgrid dispatch instance."""

    n: int
    graph: nx.Graph
    power: np.ndarray  # kW each asset injects when on
    cost: np.ndarray  # INR for running the asset over the interval
    priority: np.ndarray  # 1.0 for assets sited at critical loads, else 0
    deficit: float  # kW that must be covered
    line_stress: dict[tuple[int, int], float]  # overload (in deficit units) if both ends inject
    capacity: dict[tuple[int, int], float] = field(default_factory=dict)  # kW per line
    kinds: list[str] = field(default_factory=list)
    assets: list[str] = field(default_factory=list)
    seed: int | None = None

    @property
    def density(self) -> float:
        return nx.density(self.graph)


def make_microgrid(
    n: int = 8,
    density: float = 0.4,
    seed: int | None = 0,
    deficit_fraction: float = 0.45,
    interval_h: float = 1.0,
    line_capacity_kw: float = 8.0,
) -> Microgrid:
    """Generate a reproducible synthetic microgrid.

    Args:
        n: number of nodes (= qubits), 6-12 for the hackathon.
        density: edge density of the distribution graph in (0, 1].
        seed: RNG seed; the same seed always yields the same instance.
        deficit_fraction: deficit as a share of total dispatchable power, so the
            balance constraint is neither trivial nor impossible.
        interval_h: dispatch interval length in hours (cost = kW * h * INR/kWh).
        line_capacity_kw: nominal thermal capacity of each distribution line.
    """
    if not 2 <= n <= 20:
        raise ValueError("n must be in [2, 20]")
    if not 0 < density <= 1:
        raise ValueError("density must be in (0, 1]")
    rng = np.random.default_rng(seed)

    graph = _connected_graph(n, density, rng)

    kinds = [NODE_KINDS[i % len(NODE_KINDS)] for i in rng.permutation(n)]
    asset_names = list(ASSET_TYPES)
    assets = [asset_names[i] for i in rng.choice(len(asset_names), size=n, p=[0.45, 0.25, 0.30])]

    power = np.array([rng.uniform(*ASSET_TYPES[a]["power"]) for a in assets]).round(1)
    unit_cost = np.array([rng.uniform(*ASSET_TYPES[a]["cost"]) for a in assets])
    cost = (power * unit_cost * interval_h).round(1)
    priority = np.array([1.0 if k in CRITICAL_KINDS else 0.0 for k in kinds])

    deficit = float(round(deficit_fraction * power.sum(), 1))
    # Shared feeder lines have a thermal capacity (kW). If two neighbouring
    # assets inject together beyond it, the line overloads. line_stress is the
    # overload in units of the deficit (0 = the pair is always safe).
    capacity = {(int(i), int(j)): float(round(line_capacity_kw * rng.uniform(0.9, 1.1), 2)) for i, j in graph.edges}
    line_stress = {
        e: float(round(max(0.0, power[e[0]] + power[e[1]] - cap) / deficit, 4)) for e, cap in capacity.items()
    }
    return Microgrid(
        n=n,
        graph=graph,
        power=power,
        cost=cost,
        priority=priority,
        deficit=deficit,
        line_stress=line_stress,
        capacity=capacity,
        kinds=kinds,
        assets=assets,
        seed=seed,
    )


def _connected_graph(n: int, density: float, rng: np.random.Generator) -> nx.Graph:
    """Random connected graph with approximately the requested edge density.

    Starts from a random spanning tree (a radial feeder, typical of rural grids)
    and adds random extra edges until the target density is reached.
    """
    order = rng.permutation(n)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for k in range(1, n):
        g.add_edge(int(order[k]), int(order[rng.integers(0, k)]))
    target = max(n - 1, int(round(density * n * (n - 1) / 2)))
    candidates = [(i, j) for i in range(n) for j in range(i + 1, n) if not g.has_edge(i, j)]
    rng.shuffle(candidates)
    for i, j in candidates[: max(0, target - g.number_of_edges())]:
        g.add_edge(int(i), int(j))
    return g


def daily_profile(hours: int = 24, seed: int | None = 0) -> dict[str, np.ndarray]:
    """Synthetic 24-h village demand / solar / wind profile in kW.

    Used for the deployment-day experiment: each hour with a positive net
    deficit becomes one dispatch instance.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(hours)
    demand = 6 + 4 * np.exp(-((t - 8) ** 2) / 6) + 9 * np.exp(-((t - 20) ** 2) / 5)
    demand += rng.normal(0, 0.4, hours)
    solar = np.clip(14 * np.sin(np.pi * (t - 6) / 12), 0, None) * rng.uniform(0.8, 1.0, hours)
    wind = np.clip(2.5 + rng.normal(0, 1.2, hours), 0, None)
    return {"hour": t, "demand": demand.round(2), "solar": solar.round(2), "wind": wind.round(2)}


# ---------------------------------------------------------------------------
# Village master data: sites with their own hourly demand, assets, lines and a
# renewable profile. Stored as plain CSV files under data/villages/<name>/ so
# anyone can open, inspect or edit them. Dispatch instances are derived from it:
#     village deficit(h) = sum_sites demand(h) - solar(h) - wind(h)
# ---------------------------------------------------------------------------

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

# Peak demand range (kW) per site type.
PEAK_DEMAND_KW = {
    "household": (1.0, 2.0),
    "school": (1.5, 3.0),
    "water_pump": (2.0, 3.5),
    "cold_store": (1.5, 2.5),
    "clinic": (1.5, 3.0),
}


def _bump(t: np.ndarray, centre: float, width: float) -> np.ndarray:
    return np.exp(-((t - centre) ** 2) / (2 * width**2))


def load_shape(kind: str, hours: int = 24) -> np.ndarray:
    """Per-unit (0-1) daily load shape of a site type; multiply by peak kW."""
    t = np.arange(hours, dtype=float)
    if kind == "household":  # lights + cooking: small morning, big evening peak
        s = 0.25 + 0.35 * _bump(t, 7, 1.5) + 0.75 * _bump(t, 20, 1.8)
    elif kind == "school":  # daytime only
        s = 0.05 + 0.9 * ((t >= 9) & (t <= 16))
    elif kind == "water_pump":  # pumping before and after the working day
        s = 0.1 + 0.9 * _bump(t, 6, 1.2) + 0.9 * _bump(t, 18.5, 1.3)
    elif kind == "cold_store":  # compressors run all day, harder in the heat
        s = 0.65 + 0.3 * _bump(t, 14, 3.0)
    elif kind == "clinic":  # vaccine fridge 24/7, consultations by day, evening lights
        s = 0.45 + 0.5 * ((t >= 9) & (t <= 17)) + 0.35 * _bump(t, 20, 1.5)
    else:
        raise ValueError(f"unknown site type {kind!r}")
    return s / s.max()


@dataclass
class Village:
    """Master data for one village microgrid."""

    name: str
    sites: pd.DataFrame  # one row per site / flexible asset
    lines: pd.DataFrame  # distribution lines with capacity
    demand: pd.DataFrame  # hourly demand per site (kW), columns = site_id
    renewables: pd.DataFrame  # hourly solar and wind output for the village (kW)
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.sites)

    def hourly_balance(self) -> pd.DataFrame:
        """Village-level demand, renewables and the deficit to dispatch, per hour."""
        total = self.demand.sum(axis=1)
        df = pd.DataFrame({
            "hour": self.renewables["hour"].to_numpy(),
            "demand_kw": total.to_numpy(),
            "solar_kw": self.renewables["solar_kw"].to_numpy(),
            "wind_kw": self.renewables["wind_kw"].to_numpy(),
        })
        df["net_deficit_kw"] = (df.demand_kw - df.solar_kw - df.wind_kw).round(2)
        df["dispatch_needed"] = df.net_deficit_kw > 0.5
        df["coverable"] = df.net_deficit_kw <= self.sites.asset_kw.sum()
        return df.round(2)

    def microgrid(self, hour: int, interval_h: float = 1.0) -> Microgrid:
        """Dispatch instance for one hour, with the deficit taken from the data."""
        bal = self.hourly_balance().set_index("hour")
        deficit = float(round(bal.loc[hour, "net_deficit_kw"], 1))
        if deficit <= 0:
            raise ValueError(f"no deficit at hour {hour}: renewables cover demand")
        s = self.sites.sort_values("site_id")
        power = s.asset_kw.to_numpy(dtype=float)
        cost = (power * s.price_inr_per_kwh.to_numpy(dtype=float) * interval_h).round(1)
        g = nx.Graph()
        g.add_nodes_from(range(self.n))
        capacity = {}
        for r in self.lines.itertuples():
            e = (min(int(r.from_site), int(r.to_site)), max(int(r.from_site), int(r.to_site)))
            g.add_edge(*e)
            capacity[e] = float(r.capacity_kw)
        stress = {e: float(round(max(0.0, power[e[0]] + power[e[1]] - c) / deficit, 4)) for e, c in capacity.items()}
        return Microgrid(
            n=self.n, graph=g, power=power, cost=cost,
            priority=s.critical.to_numpy(dtype=float), deficit=deficit,
            line_stress=stress, capacity=capacity,
            kinds=s.site_type.tolist(), assets=s.asset.tolist(), seed=self.meta.get("seed"),
        )

    # ---- persistence ------------------------------------------------------
    def save(self, folder: str | Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.sites.to_csv(folder / "sites.csv", index=False)
        self.lines.to_csv(folder / "lines.csv", index=False)
        d = self.demand.copy()
        d.columns = [f"site_{c}_kw" for c in d.columns]
        d.insert(0, "hour", self.renewables["hour"].to_numpy())
        d.to_csv(folder / "site_demand.csv", index=False)
        self.renewables.to_csv(folder / "renewables.csv", index=False)
        (folder / "village.json").write_text(json.dumps({"name": self.name, **self.meta}, indent=2))
        return folder


def load_village(folder: str | Path) -> Village:
    """Load village master data written by Village.save (or edited by hand)."""
    folder = Path(folder)
    meta = json.loads((folder / "village.json").read_text())
    sites = pd.read_csv(folder / "sites.csv")
    sites["critical"] = sites["critical"].astype(bool)
    d = pd.read_csv(folder / "site_demand.csv")
    demand = d.drop(columns="hour")
    demand.columns = [int(c.split("_")[1]) for c in demand.columns]
    return Village(
        name=meta.pop("name"), sites=sites, lines=pd.read_csv(folder / "lines.csv"),
        demand=demand, renewables=pd.read_csv(folder / "renewables.csv"), meta=meta,
    )


def make_village(
    n: int = 8, density: float = 0.4, seed: int = 0, name: str | None = None,
    peak_deficit_fraction: float = 0.45,
) -> Village:
    """Synthetic village master data.

    Sites, assets and lines are the same as make_microgrid(n, density, seed), so
    a village and the benchmark instance with the same seed share their assets.
    On top of that it adds per-site hourly demand and a solar + wind profile.
    """
    g = make_microgrid(n, density, seed=seed)
    rng = np.random.default_rng(10_000 + seed)
    price = g.cost / g.power  # INR per kWh (1-h interval)
    peaks = np.array([rng.uniform(*PEAK_DEMAND_KW[k]) for k in g.kinds]).round(2)
    sites = pd.DataFrame({
        "site_id": range(n),
        "site_name": [f"{k.replace('_', ' ').title()} {i}" for i, k in enumerate(g.kinds)],
        "site_type": g.kinds,
        "critical": g.priority.astype(bool),
        "peak_demand_kw": peaks,
        "asset": g.assets,
        "asset_kw": g.power,
        "price_inr_per_kwh": price.round(3),
    })
    lines = pd.DataFrame(
        [{"from_site": i, "to_site": j, "capacity_kw": g.capacity[(min(i, j), max(i, j))]} for i, j in g.graph.edges]
    )
    hours = np.arange(24)
    demand = pd.DataFrame({i: (peaks[i] * load_shape(k) * rng.uniform(0.92, 1.08, 24)).round(2)
                           for i, k in enumerate(g.kinds)})
    clouds = rng.uniform(0.75, 1.0, 24)
    sun = np.clip(np.sin(np.pi * (hours - 6) / 12), 0, None) * clouds
    gust = rng.normal(0, 0.6, 24)
    # Scale demand (and with it PV and wind sizing) so the worst evening hour
    # needs about `peak_deficit_fraction` of all flexible capacity: hard enough
    # to be worth optimising, always coverable.
    target = peak_deficit_fraction * float(g.power.sum())
    base = demand.sum(axis=1).to_numpy()
    lo, hi = 0.1, 50.0
    for _ in range(60):
        k = (lo + hi) / 2
        net = k * base - sun * 1.3 * k * peaks.sum() - np.clip(0.12 * k * peaks.sum() + gust, 0, None)
        lo, hi = (k, hi) if net.max() < target else (lo, k)
    k = (lo + hi) / 2
    peaks = (peaks * k).round(2)
    sites["peak_demand_kw"] = peaks
    demand = (demand * k).round(2)
    solar_cap = round(1.3 * peaks.sum(), 1)  # PV sized to beat midday demand
    solar = sun * solar_cap
    wind = np.clip(0.12 * peaks.sum() + gust, 0, None)
    renewables = pd.DataFrame({"hour": hours, "solar_kw": solar.round(2), "wind_kw": wind.round(2)})
    meta = {"seed": seed, "density": density, "solar_capacity_kw": solar_cap,
            "peak_deficit_fraction": peak_deficit_fraction,
            "note": "Synthetic data generated by gridq.data.make_village. Edit the CSVs to use your own figures."}
    return Village(name or f"village_n{n}_s{seed}", sites, lines, demand, renewables, meta)
