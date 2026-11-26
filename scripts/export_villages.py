"""Regenerate the village master-data files in data/villages/.

    python scripts/export_villages.py

Each village folder contains:
  providers.csv   the flexible power providers: site, type (battery / diesel /
                  demand response), capacity (kW) and price (INR per kWh)
  forecast.csv    24-hour village forecast: total demand, solar and wind (kW)
  village.json    name, seed and notes

shortfall(hour) = demand - solar - wind; the providers must cover it.
The files are synthetic (hackathon rule: no real data). You can edit them by
hand; the dashboard and gridq.data.load_village read whatever is in the folder.
"""

from pathlib import Path

from gridq.data import make_village

ROOT = Path(__file__).resolve().parents[1] / "data" / "villages"
VILLAGES = [(6, 0), (8, 0), (8, 1), (10, 0), (12, 0)]

if __name__ == "__main__":
    for n, seed in VILLAGES:
        v = make_village(n, seed=seed)
        path = v.save(ROOT / v.name)
        b = v.hourly_balance()
        print(f"{path.relative_to(ROOT.parents[1])}: {n} providers, peak demand {b.demand_kw.max():.1f} kW, "
              f"peak shortfall {b.shortfall_kw.max():.1f} kW at {int(b.loc[b.shortfall_kw.idxmax(), 'hour']):02d}:00")
