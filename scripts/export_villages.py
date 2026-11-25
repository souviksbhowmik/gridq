"""Regenerate the village master-data files in data/villages/.

    python scripts/export_villages.py

Each village folder contains:
  sites.csv         one row per site: type, critical flag, peak demand, flexible asset, kW, price
  lines.csv         distribution lines between sites and their capacity (kW)
  site_demand.csv   hourly demand of every site (kW), 24 rows
  renewables.csv    hourly village solar and wind output (kW)
  village.json      name, seed and notes

The files are synthetic (hackathon rule: no real data). You can edit them by
hand; the dashboard and gridq.data.load_village read whatever is in the folder.
"""

from pathlib import Path

from gridq.data import make_village

ROOT = Path(__file__).resolve().parents[1] / "data" / "villages"
VILLAGES = [(6, 0), (8, 0), (8, 1), (10, 0), (12, 0)]

if __name__ == "__main__":
    for n, seed in VILLAGES:
        v = make_village(n, 0.4, seed=seed)
        path = v.save(ROOT / v.name)
        b = v.hourly_balance()
        print(f"{path.relative_to(ROOT.parents[1])}: {n} sites, peak demand {b.demand_kw.max():.1f} kW, "
              f"peak deficit {b.net_deficit_kw.max():.1f} kW at {int(b.loc[b.net_deficit_kw.idxmax(), 'hour']):02d}:00")
