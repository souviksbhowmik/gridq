"""IBM Quantum execution: credentials from .env, noise-model and real-hardware samplers.

Credentials are read from environment variables (loaded from a git-ignored
.env file). Nothing secret is ever written to disk by this module.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

ROOT = Path(__file__).resolve().parents[2]


def _service():
    from qiskit_ibm_runtime import QiskitRuntimeService

    load_dotenv(ROOT / ".env")
    token = os.getenv("QISKIT_IBM_TOKEN", "").strip()
    if not token:
        raise RuntimeError("QISKIT_IBM_TOKEN is empty: fill in .env (see .env.example)")
    return QiskitRuntimeService(
        channel=os.getenv("QISKIT_IBM_CHANNEL", "ibm_quantum_platform").strip(),
        token=token,
        instance=os.getenv("QISKIT_IBM_INSTANCE", "").strip() or None,
    )


def get_backend(min_qubits: int = 12):
    """Backend named in IBM_BACKEND, or the least-busy operational one."""
    service = _service()
    name = os.getenv("IBM_BACKEND", "").strip()
    if name:
        return service.backend(name)
    return service.least_busy(operational=True, simulator=False, min_num_qubits=min_qubits)


def hardware_sampler(backend, shots: int = 4096, mitigate: bool = True):
    """Runtime SamplerV2 with dynamical decoupling + gate twirling.

    For a sampler, twirling + DD are the standard mitigation knobs; readout
    errors are additionally suppressed by measurement twirling.
    """
    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = shots
    if mitigate:
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XpXm"
        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.enable_measure = True
        sampler.options.twirling.num_randomizations = "auto"
    return sampler


def noisy_sampler(backend=None, seed: int = 0):
    """Aer sampler with the noise model of `backend`.

    If no backend is given (or no credentials), uses a bundled fake Heron
    backend snapshot so the experiment runs fully offline.
    """
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2 as AerSampler

    if backend is None:
        backend = fake_backend()
    sim = AerSimulator.from_backend(backend, seed_simulator=seed)
    return AerSampler.from_backend(sim, seed=seed) if hasattr(AerSampler, "from_backend") else AerSampler(
        options={"backend_options": {"noise_model": sim.options.noise_model}, "run_options": {"seed": seed}}
    )


def fake_backend():
    """Offline snapshot of a real IBM device (Heron r2 'ibm_fez' if available)."""
    from qiskit_ibm_runtime import fake_provider

    for name in ("FakeFez", "FakeTorino", "FakeSherbrooke"):
        if hasattr(fake_provider, name):
            return getattr(fake_provider, name)()
    raise RuntimeError("No fake backend available in qiskit_ibm_runtime.fake_provider")


def backend_pass_manager(backend, seed: int = 0):
    return generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=seed)
