"""Synthetic rural-microgrid instances (no real data is used anywhere).

One dispatch interval (e.g. the 19:00 evening peak) is modelled. The village is
one shared power pool ("copper plate": no line limits). Each site owns one
flexible asset that can be switched on (x_i = 1) to add p_i kW to the pool:
battery discharge, a small diesel genset, or a demand-response contract (the
site cuts its own use by p_i kW, which counts the same as supplying it).
The operator must cover the village deficit D = demand - solar - wind.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# name, power range (kW), cost per kWh range (INR), share of nodes
ASSET_TYPES = {
    "battery": dict(power=(2.0, 5.0), cost=(4.0, 7.0)),
    "diesel": dict(power=(4.0, 8.0), cost=(22.0, 30.0)),
    "demand_response": dict(power=(1.0, 3.0), cost=(10.0, 14.0)),
}
NODE_KINDS = ["household", "household", "household", "school", "water_pump", "cold_store", "clinic"]


@dataclass
class Microgrid:
    """A single microgrid dispatch instance."""

    n: int
    power: np.ndarray  # kW each asset adds to the village pool when on
    cost: np.ndarray  # INR for running the asset over the interval
    deficit: float  # kW that must be covered
    kinds: list[str] = field(default_factory=list)
    assets: list[str] = field(default_factory=list)
    seed: int | None = None


def make_microgrid(
    n: int = 8,
    seed: int | None = 0,
    deficit_fraction: float = 0.45,
    interval_h: float = 1.0,
) -> Microgrid:
    """Generate a reproducible synthetic microgrid.

    Args:
        n: number of sites (= qubits), 6-12 for the hackathon.
        seed: RNG seed; the same seed always yields the same instance.
        deficit_fraction: deficit as a share of total flexible power, so the
            balance constraint is neither trivial nor impossible.
        interval_h: dispatch interval length in hours (cost = kW * h * INR/kWh).
    """
    if not 2 <= n <= 20:
        raise ValueError("n must be in [2, 20]")
    rng = np.random.default_rng(seed)
    # Earlier versions drew a distribution-line graph first. The draw is kept
    # (and discarded) so every seed still yields exactly the same sites and
    # assets as before, e.g. the instance run on ibm_fez (n=6, seed=0).
    _legacy_graph_draw(n, rng)

    kinds = [NODE_KINDS[i % len(NODE_KINDS)] for i in rng.permutation(n)]
    asset_names = list(ASSET_TYPES)
    assets = [asset_names[i] for i in rng.choice(len(asset_names), size=n, p=[0.45, 0.25, 0.30])]

    power = np.array([rng.uniform(*ASSET_TYPES[a]["power"]) for a in assets]).round(1)
    unit_cost = np.array([rng.uniform(*ASSET_TYPES[a]["cost"]) for a in assets])
    cost = (power * unit_cost * interval_h).round(1)

    deficit = float(round(deficit_fraction * power.sum(), 1))
    return Microgrid(n=n, power=power, cost=cost, deficit=deficit, kinds=kinds, assets=assets, seed=seed)


def _legacy_graph_draw(n: int, rng: np.random.Generator, density: float = 0.4) -> None:
    """Consume random numbers exactly as the old line-graph generator did (output unused)."""
    rng.permutation(n)
    for k in range(1, n):
        rng.integers(0, k)
    n_tree = n - 1
    target = max(n_tree, int(round(density * n * (n - 1) / 2)))
    rng.shuffle(list(range(n * (n - 1) // 2 - n_tree)))
    del target


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
# Village master data. Stored as plain CSV files under data/villages/<name>/:
#     providers.csv   the flexible power providers (one per site)
#     forecast.csv    hourly village demand, solar and wind forecast
# The village is one shared pool, so only the village total demand matters:
#     shortfall(h) = demand(h) - solar(h) - wind(h)
# ---------------------------------------------------------------------------

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

# Typical peak demand (kW) per site type. Only used to synthesise a realistic
# village-total demand curve; individual site demand is not stored or used.
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
    """Per-unit (0-1) daily load shape of a site type; used to build the village curve."""
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
    providers: pd.DataFrame  # one row per flexible provider: kW and price
    forecast: pd.DataFrame  # hourly village demand_kw, solar_kw, wind_kw
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.providers)

    def hourly_balance(self) -> pd.DataFrame:
        """Village demand, renewables and the shortfall to dispatch, per hour."""
        df = self.forecast[["hour", "demand_kw", "solar_kw", "wind_kw"]].copy()
        df["renewables_kw"] = df.solar_kw + df.wind_kw
        df["shortfall_kw"] = (df.demand_kw - df.renewables_kw).round(2)
        df["dispatch_needed"] = df.shortfall_kw > 0.5
        df["coverable"] = df.shortfall_kw <= self.providers.capacity_kw.sum()
        return df.round(2)

    def microgrid(self, hour: int, interval_h: float = 1.0) -> Microgrid:
        """Dispatch instance for one hour: cover that hour's shortfall."""
        bal = self.hourly_balance().set_index("hour")
        deficit = float(round(bal.loc[hour, "shortfall_kw"], 1))
        if deficit <= 0:
            raise ValueError(f"no shortfall at hour {hour}: renewables cover demand")
        p = self.providers.sort_values("provider_id")
        power = p.capacity_kw.to_numpy(dtype=float)
        cost = (power * p.price_inr_per_kwh.to_numpy(dtype=float) * interval_h).round(1)
        return Microgrid(
            n=self.n, power=power, cost=cost, deficit=deficit,
            kinds=p.site_type.tolist(), assets=p.provider_type.tolist(), seed=self.meta.get("seed"),
        )

    # ---- persistence ------------------------------------------------------
    def save(self, folder: str | Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.providers.to_csv(folder / "providers.csv", index=False)
        self.forecast.to_csv(folder / "forecast.csv", index=False)
        (folder / "village.json").write_text(json.dumps({"name": self.name, **self.meta}, indent=2))
        return folder


def load_village(folder: str | Path) -> Village:
    """Load village master data written by Village.save (or edited by hand)."""
    folder = Path(folder)
    meta = json.loads((folder / "village.json").read_text())
    return Village(
        name=meta.pop("name"),
        providers=pd.read_csv(folder / "providers.csv"),
        forecast=pd.read_csv(folder / "forecast.csv"),
        meta=meta,
    )


def make_village(
    n: int = 8, seed: int = 0, name: str | None = None,
    peak_deficit_fraction: float = 0.45,
) -> Village:
    """Synthetic village master data.

    Providers are the same as make_microgrid(n, seed), so a village and the
    benchmark instance with the same seed share their providers. The village
    demand curve is synthesised from typical daily patterns of the site types
    and scaled so the worst evening shortfall needs about
    `peak_deficit_fraction` of all flexible capacity (hard, but coverable).
    """
    g = make_microgrid(n, seed=seed)
    rng = np.random.default_rng(10_000 + seed)
    price = g.cost / g.power  # INR per kWh (1-h interval)
    peaks = np.array([rng.uniform(*PEAK_DEMAND_KW[k]) for k in g.kinds]).round(2)
    providers = pd.DataFrame({
        "provider_id": range(n),
        "site_name": [f"{k.replace('_', ' ').title()} {i}" for i, k in enumerate(g.kinds)],
        "site_type": g.kinds,
        "provider_type": g.assets,
        "capacity_kw": g.power,
        "price_inr_per_kwh": price.round(3),
    })
    hours = np.arange(24)
    site_curves = np.array([(peaks[i] * load_shape(k) * rng.uniform(0.92, 1.08, 24)).round(2)
                            for i, k in enumerate(g.kinds)])
    clouds = rng.uniform(0.75, 1.0, 24)
    sun = np.clip(np.sin(np.pi * (hours - 6) / 12), 0, None) * clouds
    gust = rng.normal(0, 0.6, 24)
    target = peak_deficit_fraction * float(g.power.sum())
    base = site_curves.sum(axis=0)
    lo, hi = 0.1, 50.0
    for _ in range(60):
        k = (lo + hi) / 2
        net = k * base - sun * 1.3 * k * peaks.sum() - np.clip(0.12 * k * peaks.sum() + gust, 0, None)
        lo, hi = (k, hi) if net.max() < target else (lo, k)
    k = (lo + hi) / 2
    peaks = (peaks * k).round(2)
    demand = (site_curves * k).round(2).sum(axis=0)
    solar_cap = round(1.3 * peaks.sum(), 1)  # PV sized to beat midday demand
    solar = sun * solar_cap
    wind = np.clip(0.12 * peaks.sum() + gust, 0, None)
    forecast = pd.DataFrame({"hour": hours, "demand_kw": demand.round(2),
                             "solar_kw": solar.round(2), "wind_kw": wind.round(2)})
    meta = {"seed": seed, "solar_capacity_kw": solar_cap,
            "peak_deficit_fraction": peak_deficit_fraction,
            "note": "Synthetic data generated by gridq.data.make_village. Edit the CSVs to use your own figures."}
    return Village(name or f"village_n{n}_s{seed}", providers, forecast, meta)
