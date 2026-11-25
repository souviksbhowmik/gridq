# QFF 2026 Hackathon: Project Plan

**Track 2: Climate, Energy & Agriculture**
**Project:** QAOA load balancing for rural microgrids (working name: *GridQ*)

---

## 1. Key facts from the requirement documents

| Item | Detail |
|---|---|
| Hacking window | Tue 24 Nov 19:00 → **Thu 26 Nov 23:59 IST (code freeze)** |
| Team lock | 19:00 Tue (run-of-show); repo shared with mentor by midnight Tue |
| Midpoint review | **Thu 26 Nov 16:30 IST**: top 20 teams, 15-min mentor sprint |
| Finalists announced | Fri 27 Nov 08:00 (top 10) |
| Grand finals | Fri 27 Nov 09:30: 5-min live pitch + 3-min Q&A |
| Mentors (Climate) | Bharadwaj Mummaneni, Phalak Bhatnagar; hardware: Guncha Malik |
| Channel | `#challenge-climate` on discord.gg/u2MKXRT4n |
| Resources | Free IBM Quantum credits per registered team (real hardware access) |
| Data rule | Synthetic or public, anonymised data only |

**Required deliverables for Track 2:**
- QUBO formulation
- Convergence vs graph density
- Comparison against greedy and simulated-annealing baselines
- Deployment metrics: % demand met and cost saved
- **Stretch goal:** run at least one instance on real IBM hardware and report how noise changes solution quality

**Scoring:**
- 30% Technical: clean, reproducible code, circuit depth, Runtime primitives, error mitigation, tests, README
- 25% Impact
- 25% Innovation and honest assessment of quantum advantage
- 20% Pitch
- Ties are broken by the Technical score, then the Impact score.

---

## 2. Problem formulation

### Scenario
A rural microgrid has N = 6–12 nodes (households, a clinic, a school, a water pump, a cold store). Each node has:
- Solar and wind generation, simulated with synthetic profiles
- Demand
- Optionally, a battery or backup asset

The nodes are connected by a graph of distribution lines with limited capacity. In each dispatch interval (for example, the evening peak), the operator decides which flexible assets to dispatch so that supply meets demand at the lowest cost, without overloading shared lines.

### Decision variables
x_i ∈ {0,1}: whether flexible asset i is dispatched (battery discharge, backup generator, or a deferrable load shifted). This uses **one qubit per node**, so 6–12 qubits.

### QUBO

```
H(x) = A · ( Σ_i p_i x_i − D_net )²           # balance: meet the net deficit
     + Σ_i c_i x_i                             # dispatch cost (diesel ≫ battery)
     + B · Σ_(i,j)∈E w_ij x_i x_j              # line congestion between neighbours
     − C · Σ_i r_i x_i   (optional)            # priority for critical loads (clinic)
```

- The balance term couples every pair of nodes (dense).
- The congestion term follows the grid graph, so **graph density** directly changes the structure of the problem. This is the deliverable axis "convergence vs graph density".
- Penalty weights A and B are calibrated so that the brute-force optimum is always feasible. They are tuned once, documented, and covered by a unit test.
- The QUBO is built with `qiskit_optimization.QuadraticProgram`, converted to an Ising operator, and fed to QAOA.

### Metrics reported for every instance
- **Approximation ratio** vs brute-force optimum (exact up to 2¹² = 4096 states)
- **% demand met**
- **Cost saved vs a naive policy** (e.g. "always run diesel")
- **Probability of sampling the optimal bitstring**
- **Circuit depth and 2-qubit gate count**, before and after transpilation to the heavy-hex topology
- **Runtime**

---

## 3. Solutions to compare

| # | Solver | Role |
|---|---|---|
| 1 | Brute force | Ground truth (feasible at ≤12 variables) |
| 2 | Greedy (cheapest-per-kW first, respecting lines) | Required baseline |
| 3 | Simulated annealing (`dwave-neal` or a hand-written version) | Required baseline |
| 4 | QAOA, ideal statevector, p = 1…4 | Quantum best case |
| 5 | QAOA, Aer noise model from a real backend | Realistic expectation |
| 6 | QAOA, real IBM hardware (Heron), 1–2 instances | Stretch goal, with error mitigation |

**QAOA details:**
- Built with `QAOAAnsatz` and run through the `SamplerV2` primitive.
- Parameters are optimised with COBYLA. Each depth p starts from the parameters found at p−1 (warm start, or INTERP-style initialisation).
- On hardware, the following are used:
  - `optimization_level=3` transpile
  - Dynamical decoupling and Pauli twirling through Runtime options
  - Measurement error mitigation
  - **CVaR objective**, which is robust to noise

**Honest-advantage framing (for the 25% Innovation criterion):** at this size, classical solvers win. The value of the project is showing:
- how QAOA solution quality scales with p and graph density,
- what noise costs, and
- where the crossover would need to occur.

The README and pitch both state this explicitly.

---

## 4. Experiments

1. **Correctness:** for 6 nodes, all solvers vs brute force, across 20 random instances.
2. **Depth sweep:** p = 1, 2, 3, 4 at N = 8. Plot approximation ratio and P(optimal) against p.
3. **Graph-density sweep:** edge density 0.2 → 1.0 at N = 8 and N = 10. Plot convergence iterations, approximation ratio, and transpiled 2-qubit depth (the required deliverable).
4. **Scaling:** N = 6, 8, 10, 12 for all solvers. Plot runtime and quality.
5. **Noise:** ideal vs noise model vs real hardware for one N = 6 instance, with and without error mitigation.
6. **Deployment story:** a 24-hour synthetic day split into dispatch intervals, comparing % demand met and cost against a naive policy.

---

## 5. Repository layout

```
qff_hackathon/
├── README.md                  # problem, approach, how to run, results table, limitations
├── LICENSE                    # Apache-2.0 (same as Qiskit)
├── environment.yml            # conda env definition
├── requirements.txt           # pinned pip versions (exact ==)
├── src/gridq/
│   ├── data.py                # synthetic microgrid generator (seeded)
│   ├── qubo.py                # QuadraticProgram / Ising builder, penalty calibration
│   ├── classical.py           # brute force, greedy, simulated annealing
│   ├── qaoa.py                # ansatz, optimiser loop, Sampler/Aer/Runtime backends
│   ├── metrics.py             # approx ratio, % demand met, cost saved, depth stats
│   └── hardware.py            # IBM Runtime job submission + error mitigation options
├── experiments/               # one script per experiment, writes results/*.csv
├── notebooks/demo.ipynb       # live-demo notebook used in the pitch
├── app/ (optional)            # Streamlit dashboard: grid map + dispatch + solver comparison
├── results/                   # CSVs, plots, hardware job IDs
├── tests/                     # pytest: QUBO correctness, feasibility, small-N vs brute force
└── docs/architecture.png      # 1-slide architecture diagram
```

---

## 6. Environment (separate conda env only)

All Python installation goes into a new conda environment named **`qff-gridq`** (Python 3.12). Nothing is installed into `base`, and none of the existing environments are touched.

```bash
conda create -n qff-gridq python=3.12 -y
conda activate qff-gridq
pip install qiskit==2.x qiskit-aer==0.18.x qiskit-ibm-runtime==0.52.x qiskit-optimization==0.7.x \
            numpy scipy pandas networkx matplotlib dwave-neal pytest jupyter streamlit
pip freeze > requirements.txt
```

These were the latest versions on PyPI at planning time:
- qiskit 2.6.2
- qiskit-aer 0.18.4
- qiskit-ibm-runtime 0.52.1
- qiskit-optimization 0.7.0

**Step 0 is to verify that qiskit-optimization 0.7.0 works with Qiskit 2.6.** If it does not:
- pin Qiskit down to a compatible version, or
- drop qiskit-optimization and build the Ising operator directly with `SparsePauliOp`. That is about 40 lines of code.

Exact versions are pinned in `requirements.txt` (a required checklist item).

The IBM Quantum API token is kept in a `.env` file or saved with `QiskitRuntimeService.save_account`. It is never committed.

---

## 7. Timeline (≈39 h left from Wed 25 Nov 09:00 IST)

| When (IST) | Work | Owner role |
|---|---|---|
| **Wed 09:00–11:00** | Conda env, repo skeleton, licence, push to GitHub, share with mentor; hello-world QAOA on 4 nodes on Aer | Quantum dev |
| Wed 09:00–12:00 | Synthetic data generator, realistic village profiles | Domain expert |
| **Wed 11:00–18:00** | QUBO builder + penalty calibration + tests; brute force, greedy, SA | Quantum dev + baseline eng |
| **Wed 14:00–22:00** | QAOA loop (p sweep, warm start), metrics module, experiments 1–2 | Quantum dev |
| Wed 18:00–24:00 | Experiments 3–4 (density, scaling) and first plots | Baseline eng |
| **Thu 08:00–12:00** | Noise-model run, then queue **real hardware** job(s) early (queues can be long); experiment 5 | Quantum dev |
| Thu 10:00–15:00 | Deployment-day simulation (exp 6), demo notebook / Streamlit dashboard | Baseline eng + pitch lead |
| **Thu 16:30** | **Midpoint mentor sprint**: bring plots + one open question | All |
| Thu 17:00–20:00 | Apply feedback, **feature freeze at 20:00**, README results table, architecture diagram | All |
| Thu 20:00–22:30 | Record 3-min pitch video, final test run from a clean env | Pitch lead |
| **Thu ≤ 23:00** | **Submit** repo + video link + form (one hour of buffer) | Team lead |
| Fri 08:00–09:30 | If shortlisted: rehearse the 5-min live pitch, prepare Q&A answers | Pitch lead |

---

## 8. Pitch outline (3 min video, 5 min finals)

1. **Human hook (30 s):** a village clinic loses power at 7 pm while a solar battery two houses away sits full.
2. **Problem → QUBO (30 s):** show the grid graph, then the one-line Hamiltonian.
3. **Architecture (20 s):** data generator → QUBO → {classical | QAOA on Aer | IBM Heron} → metrics.
4. **Results (60 s):**
   - comparison table
   - approximation ratio vs p
   - density plot
   - ideal vs noisy vs hardware
5. **Honest take (20 s):** classical solvers win today; show what would need to change for quantum to win.
6. **Impact and next steps (20 s):** % demand met, cost saved, path to a real deployment with a community-grid operator.

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Penalty weights produce infeasible QAOA answers | Calibrate A and B against brute force; post-process with feasibility repair; report raw and repaired |
| QAOA quality collapses at 12 qubits / dense graphs | That *is* a finding: report it. Core results at N = 6–8 |
| Hardware queue too long | Submit hardware jobs Thursday morning; noise-model results as fallback (accepted in checklist) |
| Package incompatibility | Check versions in step 0; have the `SparsePauliOp` fallback ready |
| Running out of time | "Small and working beats big and broken". The Streamlit app is optional; the notebook demo is enough |

---

## 10. Open decisions for the team
- Confirm the track choice (Track 2) and team members, and assign the four roles.
- Choose the GitHub repo name and owner account, and register the team in the form.
- Get the IBM Quantum API token and the instance/CRN from the team credits.
- Live demo format: Jupyter notebook (default) or Streamlit dashboard.
