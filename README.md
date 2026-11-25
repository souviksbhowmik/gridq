# GridQ: QAOA load balancing for rural microgrids

**IBM Qiskit Fall Fest 2026 · 48-Hour Quantum for Humanity Hackathon · Track 2: Climate, Energy & Agriculture**

> At 19:00 the sun sets and village demand peaks: lights, the clinic's vaccine fridge, the water pump.
> The microgrid operator must choose which flexible assets to switch on (batteries, a diesel genset,
> demand-response contracts) to cover the shortfall **at the lowest cost, without overloading shared
> lines, and keeping critical sites powered**. Today that is done by rule of thumb, and the fallback
> is "run the diesel". GridQ formulates the decision as a QUBO, solves it with QAOA on Qiskit Runtime
> primitives, and benchmarks the result honestly against classical solvers.

![architecture](docs/architecture.png)

**Team:** _TBD_ · **Track:** Climate · **Mentors:** Bharadwaj Mummaneni, Phalak Bhatnagar · **Licence:** Apache-2.0

---

## 1. Problem

| | |
|---|---|
| Who benefits | Rural communities on solar/wind/battery microgrids, and the small operators who run them |
| Decision | For one dispatch interval: which of the *n* flexible assets (6–12) to switch on |
| Constraints | Supply ≥ deficit; shared feeder lines have a thermal capacity |
| Objective | Minimise running cost (₹), prefer supplying critical loads (clinic, water pump, cold store) |
| Data | **100% synthetic** (`src/gridq/data.py`, seeded). No real customer or utility data |

## 2. QUBO formulation (`src/gridq/qubo.py`)

One binary variable per asset, so **one qubit per node**.

$$
H(x)=A\Big(\sum_i \tfrac{p_i}{D}\,x_i-1\Big)^2+\sum_i \tilde c_i\,x_i+B\!\!\sum_{(i,j)\in E}\! w_{ij}\,x_ix_j-C\sum_i r_i\,x_i
$$

| Term | Meaning | Weight |
|---|---|---|
| balance | injected power $p_i$ should cover the deficit $D$ | A = 20 |
| cost | normalised running cost $\tilde c_i$ | 1 |
| line overload | $w_{ij} = \max(0,\,p_i+p_j-\text{cap}_{ij})/D$: penalises neighbours whose joint injection exceeds the line capacity | B = 1 |
| critical loads | $r_i = 1$ if the asset sits at a clinic, water pump or cold store | C = 0.04 |

The QUBO is built as a `qiskit_optimization.QuadraticProgram`, converted to an Ising `SparsePauliOp`, and normalised so QAOA angles transfer across instances.

**Feasibility repair.** A squared penalty cannot encode `supply ≥ deficit` exactly. So **every solver's output, quantum or classical**, goes through the same repair step (`classical.repair`): add the cheapest-per-kW asset while short, then trim waste. This keeps the comparison like-for-like.

## 3. Methods

| Solver | File | Notes |
|---|---|---|
| **Exact** (constrained brute force) | `classical.py` | Ground truth; 2¹² = 4096 states is trivial |
| **Operator rule** | `classical.py` | Cheapest-per-kW first until covered (what operators do by hand) |
| **Greedy descent** | `classical.py` | Best single bit-flip on the QUBO until stuck |
| **Simulated annealing** | `classical.py` | Metropolis, geometric schedule, 500 sweeps × 5 restarts |
| **Random sampling** | `classical.py` | *Control:* uniform bitstrings + same repair, same shot budget as QAOA |
| **QAOA** | `qaoa.py` | `QAOAAnsatz(p)`, **CVaR(α = 0.25)** objective, COBYLA, p = 1 grid-search initialisation, **INTERP** warm start between depths, `SamplerV2` |

**QAOA execution path.**
1. Angles are trained noise-free on an exact simulator. This is `fast_qaoa_probs`, a NumPy re-implementation of the ansatz that is about 100× faster than building circuits. A unit test checks it against `qiskit.quantum_info.Statevector` to within 1e-10.
2. The trained circuit is transpiled at optimisation level 3.
3. It is then sampled on one of:
   - Aer ideal
   - Aer with the noise model of an IBM Heron r2 snapshot (`FakeFez`)
   - **real IBM hardware** via `qiskit_ibm_runtime.SamplerV2`, with dynamical decoupling (XpXm) plus gate and measurement twirling

## 4. Results

Reproduce with `python experiments/run_benchmarks.py` (≈ 4 min on a laptop). 10 random instances per setting, edge density 0.4 unless stated. Raw CSVs and plots are in [`results/`](results/).

### 4.1 Solver comparison (`results/comparison_table.csv`)

`quality` = 1 for the exact optimum and 0 for the worst feasible dispatch. `optimal` = share of instances where the exact optimum was found. `saved` = cost saving vs a *diesel-first* policy.

| Solver | n = 6 quality / optimal | n = 8 | n = 10 | n = 12 quality | saved (n = 8) | runtime (n = 10) |
|---|---|---|---|---|---|---|
| Exact | 1.000 / 100% | 1.000 / 100% | 1.000 / 100% | 1.000 | 59.9% | **0.1 ms** |
| **QAOA p = 3** (ideal, 256 shots) | 1.000 / 100% | 1.000 / 100% | 1.000 / 100% | 1.000 | 59.9% | 496 ms |
| Random sampling (256 shots) | 1.000 / 100% | 1.000 / 100% | 1.000 / 100% | 1.000 | 59.9% | 19 ms |
| Simulated annealing | 0.983 / 80% | 0.958 / 40% | 0.959 / 10% | 0.981 | 58.0% | 46 ms |
| Operator rule | 0.938 / 40% | 0.959 / 30% | 0.958 / 30% | 0.949 | 59.0%* | 0.01 ms |
| Greedy descent | 0.700 / 10% | 0.667 / 0% | 0.638 / 0% | 0.661 | 25.4% | 0.05 ms |

\* The operator rule sometimes saves *more* rupees than the optimum because it ignores the critical-load preference. Its quality score is lower because it gets the trade-off wrong.

### 4.2 Does the quantum circuit help? Enrichment vs depth (`results/depth_sweep.png`)

The **enrichment** is P(sampling the optimal dispatch) ÷ P(random guess).

| p | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| enrichment (n = 8, ideal) | 2.2× | 3.8× | 5.1× | 6.4× | 7.0× |
| 2-qubit depth (generic transpile) | 26 | 42 | 58 | 74 | 90 |

CVaR(α = 0.25) gives about 40–50% more enrichment than the plain expectation value at the same depth. We measured 3.8× vs 2.65× at n = 8, p = 2, and 5.2× vs 3.6× at n = 12, p = 3.

### 4.3 Convergence vs graph density (`results/density_sweep.png`)

| density | 0.2 | 0.35 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|---|
| enrichment, n = 8 (p = 2) | 3.8× | 3.8× | 4.0× | 4.1× | 4.0× |
| enrichment, n = 10 (p = 2) | 3.7× | 3.7× | 3.7× | 4.2× | 4.5× |
| 2-qubit depth, n = 8 / n = 10 | 42 / 54 | 42 / 54 | 42 / 54 | 42 / 54 | 42 / 54 |

**Finding:** the **balance penalty couples every pair of assets**, so the QUBO is all-to-all whatever the grid density. Circuit cost depends on *n*, not on line density. Density only changes coefficients, and QAOA quality is roughly flat to slightly better on denser grids. A sparser formulation, such as regional sub-balances, is the main lever for making circuits shallower.

### 4.4 Noise (`results/noise.png`): ideal vs IBM Heron noise model, n = 6

| p | ideal enrichment | noisy enrichment | 2q gates ideal → on-device |
|---|---|---|---|
| 1 | 2.21× | 1.99× | 30 → 53 |
| 2 | 2.80× | 2.15× | 60 → 105 |
| 3 | 2.97× | 2.23× | 90 → 167 |

Routing onto the heavy-hex topology adds SWAPs and roughly doubles the 2-qubit gate count. **Under noise, gains beyond p = 2 almost disappear.** Deeper circuits add as much error as they remove. Real-hardware results go in `results/hardware_*.json` (see §6).

### 4.5 Deployment day (`results/deployment_day_summary.csv`)

We dispatched one 8-asset village hour by hour over a synthetic 24-hour demand, solar and wind profile.

| Policy | Daily cost | Saved vs diesel-first | Line-overload hours |
|---|---|---|---|
| Diesel-first (status quo) | ₹4,048 | n/a | n/a |
| Operator rule | ₹1,085 | 73.2% | 0 |
| **QAOA p = 3** | **₹1,136** | **71.9%** | **0** |
| Exact optimum | ₹1,136 | 71.9% | 0 |

QAOA matched the exact optimum in every hour. Most of the economic value comes from **optimising at all**, rather than running diesel by default. The optimiser pays a little more than the operator rule because it keeps critical loads (the clinic) supplied by their own assets.

## 5. Honest assessment and limitations

- **Classical wins at this size.** For ≤ 12 assets, brute force is exact in under 1 ms. We make **no quantum-advantage claim**.
- **QAOA's signal is real but not yet decisive.** It concentrates probability on the optimum (2–7× over random, rising with p). But with 256 shots and the same repair step, **random sampling also finds the optimum** at n ≤ 12, because the search space is so small.
- **Noise caps useful depth at p ≈ 2** on today's devices for this problem.
- **Formulation gap.** The squared balance penalty plus repair can land up to 14% (mean 2.5%) of the feasible objective range away from the true optimum. Across 45 calibration instances it was exact 62% of the time (`tests/test_gridq.py`). Slack-variable encodings cost extra qubits. We tested them and they improved this only moderately.
- **Single interval.** We do not model battery state of charge across hours or AC power flow. Lines use a simple pairwise capacity model.
- **Synthetic data only**, as the rules require.

**When could quantum matter?** District-scale grids (hundreds of assets with multi-period coupling) are beyond brute force, and heuristics there degrade. Getting there needs sparser formulations (lower circuit depth), better error mitigation, and eventually fault-tolerant hardware.

**Path to deployment.** The value delivered today is an *optimiser-in-the-loop* dispatch for community microgrid operators, saving about 70% vs diesel-first in our synthetic day. The solver backend is pluggable: classical today, quantum as hardware matures.

## 6. How to run

All Python dependencies live in a **dedicated conda environment**.

### 6.1 Set up the environment (once)

```bash
conda env create -f environment.yml        # creates "qff-gridq" (Python 3.12) with pinned packages + gridq
conda activate qff-gridq
python -m ipykernel install --user --name qff-gridq --display-name "Python (qff-gridq)"   # Jupyter kernel
```

### 6.2 Run everything on simulators (no account needed)

```bash
pytest -q                                   # 24 tests, ~2 s
python experiments/run_benchmarks.py        # all 6 experiments → results/  (~4 min)
python experiments/run_benchmarks.py --quick              # smoke test, 3 seeds
python experiments/run_benchmarks.py --only density noise # selected experiments
```

### 6.3 Live demo (both formats)

| Format | Command | Notes |
|---|---|---|
| Jupyter notebook | `jupyter notebook notebooks/demo.ipynb` | Choose kernel **Python (qff-gridq)**; *Run All* takes about 1 min |
| Streamlit dashboard | `streamlit run app/streamlit_app.py` | Opens http://localhost:8501. Pick nodes, density, p, shots, and ideal vs noisy backend, then click **Solve dispatch** |

### 6.4 Real IBM Quantum hardware (optional; uses credits)

1. Sign in at **https://quantum.cloud.ibm.com** (free IBMid).
2. **API key:** on the dashboard, click *Create API key*. It is shown only once, so copy it.
3. **Instance CRN:** go to *Instances* and copy the CRN (`crn:v1:bluemix:...`). Use the hackathon-credit instance if you have been invited to one; otherwise use the free Open plan (about 10 min of QPU time per month, which is plenty).
4. **Backend:** the *Compute resources* page lists your QPUs. Each QPU is a Qiskit backend. Our allocation is all Heron r2 with 156 qubits: `ibm_fez`, `ibm_kingston`, `ibm_marrakesh`. We use **`ibm_fez`**, because our noise-model results use `FakeFez`, a snapshot of the same device.
5. Fill in the git-ignored `.env` file:

   ```bash
   cp .env.example .env
   ```
   ```ini
   QISKIT_IBM_TOKEN=<API key>
   QISKIT_IBM_INSTANCE=<instance CRN>
   QISKIT_IBM_CHANNEL=ibm_quantum_platform
   IBM_BACKEND=ibm_fez          # empty = least-busy available QPU
   ```
6. Run:

   ```bash
   python experiments/run_hardware.py --dry-run          # free: checks credentials, prints transpiled depth
   python experiments/run_hardware.py --n 6 --reps 2     # submits 2 jobs: unmitigated + mitigated (DD + twirling)
   ```
   Options: `--n` nodes (6–12), `--reps` QAOA layers, `--seed` scenario, `--density`, `--shots` (default 4096).

Results are saved to `results/hardware_<backend>_n<n>_p<reps>_s<seed>.json`. Each file holds the job IDs, counts, enrichment, P(optimal), dispatch metrics and circuit stats. Following the rules, the script trains on the simulator first and only sends the final circuit to the QPU.

> **Security:** only the file named exactly `.env` is read by the code. All `.env*` files except `.env.example` are git-ignored. Never paste the API key into code, notebooks, issues or chat.

### 6.5 Troubleshooting

| Symptom | Fix |
|---|---|
| `QISKIT_IBM_TOKEN is empty` | Fill in `.env` (not `.env.example`) |
| `No backend` / authentication error | Check the CRN matches the instance that owns the QPU; set `IBM_BACKEND` to a name listed on *Compute resources* |
| Notebook can't find `gridq` | Select the **Python (qff-gridq)** kernel |
| Streamlit crashes on macOS | Run from the `qff-gridq` env; the app pre-imports `qiskit_aer` on the main thread to avoid a known macOS thread crash |

## 7. Repository layout

```
src/gridq/        data.py · qubo.py · classical.py · qaoa.py · metrics.py · hardware.py
experiments/      run_benchmarks.py (6 experiments) · run_hardware.py
.env.example      template for IBM credentials (copy to .env)
notebooks/        demo.ipynb
app/              streamlit_app.py
results/          CSVs + plots (+ hardware JSON with job IDs)
tests/            test_gridq.py
docs/             architecture.png
```

## 8. Versions

Pinned in `requirements.txt`; full lock in `requirements-lock.txt`:
- qiskit 2.6.2
- qiskit-aer 0.18.4
- qiskit-ibm-runtime 0.52.1
- qiskit-optimization 0.7.0
- numpy 2.5.5
- scipy 1.18.1
- Python 3.12

## 9. Credits

Built with open-source **Qiskit**, **Qiskit Aer**, **Qiskit IBM Runtime**, **Qiskit Optimization**, NumPy, SciPy, NetworkX, pandas, Matplotlib and Streamlit. INTERP initialisation is from Zhou et al., *PRX* 10, 021067 (2020). CVaR-QAOA is from Barkoutsos et al., *Quantum* 4, 256 (2020). All code was written during the hackathon window.
