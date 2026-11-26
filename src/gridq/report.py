"""Result tables and charts for notebooks: what each solver decided."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import cost_saved_pct, dispatch_metrics, naive_policy_cost

TYPE_COLOUR = {"battery": "#43a047", "demand_response": "#1e88e5", "diesel": "#757575"}


def summary_row(name: str, grid, x, exact=None, runtime_s: float | None = None, **extra) -> dict:
    """One-line result of a dispatch decision."""
    x = np.asarray(x)
    m = dispatch_metrics(grid, x)
    row = {
        "solver": name,
        "providers on": ", ".join(str(i) for i in np.flatnonzero(x)),
        "supplied kW": m["supplied_kw"],
        "shortfall kW": grid.deficit,
        "demand met %": m["demand_met_pct"],
        "cost ₹": m["cost_inr"],
        "diesel-first ₹": round(naive_policy_cost(grid), 1),
        "saved vs diesel-first %": cost_saved_pct(grid, x),
    }
    if exact is not None:
        row["gap to optimum ₹"] = round(float(grid.cost @ x - grid.cost @ exact), 1)
        row["optimal?"] = "✅" if np.array_equal(x, exact) else "❌"
    if runtime_s is not None:
        row["runtime ms"] = round(1000 * runtime_s, 1)
    row.update(extra)
    return row


def dispatch_table(providers: pd.DataFrame, grid, x) -> pd.DataFrame:
    """Per-provider instruction: which providers to switch on, and what each costs."""
    x = np.asarray(x)
    p = providers.sort_values("provider_id").reset_index(drop=True)
    return pd.DataFrame({
        "id": p.provider_id,
        "site": p.site_name,
        "provider type": p.provider_type,
        "capacity kW": p.capacity_kw,
        "switch": np.where(x == 1, "🟢 ON", "⚪ off"),
        "kW supplied": np.where(x == 1, grid.power, 0.0),
        "cost ₹": np.where(x == 1, grid.cost, 0.0),
    })


def plot_allocation(grid, results: dict, title: str = "Providers switched on by each solver"):
    """One bar per solver: stacked kW of the providers it switched on, vs the shortfall line."""
    order = np.argsort(grid.cost / grid.power)
    names = list(results)
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(names) + 1.3))
    for row, nm in enumerate(names):
        x = np.asarray(results[nm])
        left = 0.0
        for i in order:
            if x[i]:
                ax.barh(row, grid.power[i], left=left, color=TYPE_COLOUR[grid.assets[i]], edgecolor="white")
                ax.text(left + grid.power[i] / 2, row, str(i), ha="center", va="center", fontsize=8, color="white")
                left += grid.power[i]
        ax.text(left + 0.2, row, f"{left:.1f} kW · ₹{grid.cost @ x:.0f}", va="center", fontsize=8)
    ax.axvline(grid.deficit, color="#d32f2f", ls="--", lw=2)
    ax.text(grid.deficit, -0.6, f" shortfall {grid.deficit:.1f} kW", color="#d32f2f", fontsize=8)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlabel("kW switched on (numbers = provider id; green battery · blue demand response · grey diesel)")
    ax.set_xlim(0, max(grid.deficit, max(float(np.asarray(results[n]) @ grid.power) for n in names)) * 1.35)
    ax.set_title(title)
    fig.tight_layout()
    plt.show()
