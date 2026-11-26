import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from gridq import (
    brute_force, build_qubo, constrained_optimum, dispatch_metrics, dispatch_objective, repair, greedy_descent, greedy_operator,
    make_microgrid, simulated_annealing, solve_qaoa,
)
from gridq.qaoa import _bind, build_ansatz, fast_qaoa_probs, interp_init, ising_diagonal
from gridq.qubo import all_bitstrings, bitstring_to_array, energy_landscape


@pytest.fixture(params=[(6, 0), (8, 1), (10, 2)])
def qubo(request):
    n, s = request.param
    return build_qubo(make_microgrid(n=n, seed=s))


def test_generator_is_reproducible():
    a, b = make_microgrid(8, seed=7), make_microgrid(8, seed=7)
    assert np.array_equal(a.power, b.power) and a.deficit == b.deficit and a.assets == b.assets
    assert a.n == 8 and np.isclose(a.deficit, round(0.45 * a.power.sum(), 1))


def test_hardware_instance_is_unchanged():
    """The instance run on ibm_fez (n=6, seed=0) must stay identical for reproducibility."""
    g = make_microgrid(6, seed=0)
    assert g.deficit == 11.6
    assert np.allclose(g.power, [2.3, 6.6, 3.8, 3.2, 5.0, 4.9])


def test_sparsify_qubo_reduces_problem_graph_density():
    from gridq import problem_graph_density, sparsify_qubo

    q = build_qubo(make_microgrid(8, seed=0))
    assert problem_graph_density(q) == 1.0  # balance term couples every pair
    for keep in (0.2, 0.5, 0.8):
        qs = sparsify_qubo(q, keep)
        assert abs(problem_graph_density(qs) - keep) < 0.05
        assert np.allclose(np.diag(qs.Q), np.diag(q.Q))  # linear terms untouched


def test_qubo_matches_quadratic_program(qubo):
    qp = qubo.to_quadratic_program()
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.integers(0, 2, qubo.n)
        assert np.isclose(qp.objective.evaluate(x), qubo.energy(x))


def test_ising_diagonal_is_affine_in_qubo_energy(qubo):
    d, L = ising_diagonal(qubo), energy_landscape(qubo)
    _, _, scale = qubo.to_ising()
    assert np.allclose(np.diff(d) * scale, np.diff(L))


def test_bit_ordering_round_trip():
    X = all_bitstrings(4)
    k = 0b1011
    assert np.array_equal(bitstring_to_array(format(k, "04b")), X[k])


def test_repair_makes_every_dispatch_feasible(qubo):
    g = qubo.grid
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = repair(g, rng.integers(0, 2, g.n))
        assert g.power @ x >= g.deficit - 1e-9


def test_repaired_qubo_optimum_is_near_true_optimum(qubo):
    """Calibration check: QUBO optimum + repair lands near the exact constrained optimum.

    The squared balance penalty cannot encode `supply >= deficit` exactly, so a
    formulation gap exists (measured over 45 instances: mean 2.5%, max 14% of
    the feasible objective range). Documented in README "Limitations".
    """
    g = qubo.grid
    X = all_bitstrings(g.n)
    obj = dispatch_objective(g, X)
    feas = X @ g.power >= g.deficit - 1e-9
    spread = obj[feas].max() - obj[feas].min()
    true_best = dispatch_objective(g, constrained_optimum(qubo).x)
    got = dispatch_objective(g, repair(g, brute_force(qubo).x))
    assert got <= true_best + 0.15 * spread
    assert dispatch_metrics(g, repair(g, brute_force(qubo).x))["demand_met_pct"] == 100.0


def test_classical_solvers_never_beat_brute_force(qubo):
    e = brute_force(qubo).energy
    for r in (greedy_operator(qubo), greedy_descent(qubo), simulated_annealing(qubo, sweeps=200, restarts=3)):
        assert r.energy >= e - 1e-9


def test_simulated_annealing_finds_optimum_small():
    q = build_qubo(make_microgrid(6, seed=3))
    assert np.isclose(simulated_annealing(q).energy, brute_force(q).energy)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_fast_simulator_matches_qiskit(p):
    q = build_qubo(make_microgrid(6, seed=4))
    ansatz, _ = build_ansatz(q, p)
    params = np.random.default_rng(p).uniform(-1, 1, 2 * p)
    ref = Statevector(_bind(ansatz, params)).probabilities()
    assert np.allclose(ref, fast_qaoa_probs(ising_diagonal(q), q.n, params), atol=1e-10)


def test_interp_shapes():
    assert interp_init(np.array([0.1, 0.2])).shape == (4,)
    assert interp_init(np.arange(6.0)).shape == (8,)


def test_qaoa_beats_uniform_sampling():
    q = build_qubo(make_microgrid(6, seed=5))
    r = solve_qaoa(q, reps=2, shots=2048)
    uniform_mean = energy_landscape(q).mean()
    assert r.expected_energy < uniform_mean
    assert r.best_energy >= brute_force(q).energy - 1e-9


def test_village_master_data_round_trip(tmp_path):
    from gridq.data import load_village, make_village

    v = make_village(8, seed=0)
    w = load_village(v.save(tmp_path / "v"))
    assert w.n == 8 and len(w.forecast) == 24
    assert np.allclose(v.hourly_balance().shortfall_kw, w.hourly_balance().shortfall_kw)


def test_village_shortfall_is_demand_minus_renewables():
    from gridq.data import make_village

    v = make_village(8, seed=1)
    b = v.hourly_balance()
    assert np.allclose(b.shortfall_kw, (b.demand_kw - b.solar_kw - b.wind_kw).round(2), atol=0.02)
    assert b.coverable.all()  # sized so every hour can be covered
    peak = int(b.loc[b.shortfall_kw.idxmax(), "hour"])
    assert 17 <= peak <= 21  # evening peak
    g = v.microgrid(peak)
    assert g.n == 8 and np.isclose(g.deficit, round(b.shortfall_kw.max(), 1))


def test_shipped_village_files_load():
    from pathlib import Path

    from gridq.data import load_village

    root = Path(__file__).resolve().parents[1] / "data" / "villages"
    folders = [p for p in root.iterdir() if (p / "providers.csv").exists()]
    assert folders
    for f in folders:
        v = load_village(f)
        assert v.hourly_balance().coverable.all()


def test_readable_circuit_matches_qaoa_ansatz():
    from gridq.qaoa import readable_circuit

    q = build_qubo(make_microgrid(6, seed=0))
    params = np.random.default_rng(1).uniform(-1, 1, 4)
    rc = readable_circuit(q, params, digits=12).remove_final_measurements(inplace=False)
    ref = Statevector(_bind(build_ansatz(q, 2)[0], params)).probabilities()
    assert np.allclose(Statevector(rc).probabilities(), ref, atol=1e-8)
