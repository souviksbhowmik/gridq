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
