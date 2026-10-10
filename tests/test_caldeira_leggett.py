from typing import Any

import numpy as np
import pytest
import qutip  # type: ignore lib
from qutip.solver.sode._noise import Wiener  # type: ignore lib
from slate_core import basis
from slate_core.metadata.volume import spaced_volume_metadata_from_stacked_delta_x

from slate_quantum import operator, state
from slate_quantum.dynamics import LangevinParameters
from slate_quantum.dynamics.caldeira_leggett._periodic import (
    _build_environment_operators,  # type: ignore private
    _get_periodic_basis,  # type: ignore private
)
from slate_quantum.dynamics.caldeira_leggett._util import (
    _WindowedWiener,  # type: ignore private
    windowed_wiener,
)
from slate_quantum.operator import operator_basis

type _Matrix = np.ndarray[Any, np.dtype[np.complexfloating]]

FRICTION = 0.1
"""The friction, in units where hbar = m = k_B T = 1."""


def _adjoint_lindblad(
    hamiltonian: _Matrix, collapse_operators: list[_Matrix], observable: _Matrix
) -> _Matrix:
    """Get dA/dt for the master equation, in the Heisenberg picture with hbar = 1."""
    out = 1j * (hamiltonian @ observable - observable @ hamiltonian)
    for c in collapse_operators:
        c_dag = c.conj().T
        out += c_dag @ observable @ c - 0.5 * (
            c_dag @ c @ observable + observable @ c_dag @ c
        )
    return out


def _expectation(psi: _Matrix, observable: _Matrix) -> float:
    return np.real(np.vdot(psi, observable @ psi) / np.vdot(psi, psi)).item()


def _assert_langevin_dynamics(
    psi: _Matrix,
    hamiltonian: _Matrix,
    collapse_operators: list[_Matrix],
    p: _Matrix,
    *,
    rtol: float,
) -> None:
    """Check d<p>/dt = -gamma <p> and d<p^2>/dt = -2 gamma <p^2> + 2 m gamma k_B T."""
    p_squared = p @ p
    dp = _expectation(psi, _adjoint_lindblad(hamiltonian, collapse_operators, p))
    np.testing.assert_allclose(dp, -FRICTION * _expectation(psi, p), rtol=rtol)

    dp_squared = _expectation(
        psi, _adjoint_lindblad(hamiltonian, collapse_operators, p_squared)
    )
    np.testing.assert_allclose(
        dp_squared,
        -2 * FRICTION * _expectation(psi, p_squared) + 2 * FRICTION,
        rtol=rtol,
    )


def test_caldeira_leggett_langevin_dynamics() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x((np.array([60.0]),), (512,))
    position_basis = basis.from_metadata(metadata)
    out_basis = operator_basis(position_basis)
    n = position_basis.size

    def as_matrix(op: operator.Operator[Any, Any]) -> _Matrix:
        return op.with_basis(out_basis).raw_data.reshape(n, n)

    p = as_matrix(operator.build.p(metadata, axis=0, hbar=1))
    shift = as_matrix(
        operator.build.caldeira_leggett_shift(metadata, friction=FRICTION, hbar=1)
    )
    collapse = as_matrix(
        operator.build.caldeira_leggett_collapse(
            metadata, friction=FRICTION, kb_t=1, mass=1, hbar=1
        )
    )
    psi = (
        state.build_coherent(metadata, (30.0,), (0.7,), sigma_0=(1.5,))
        .with_basis(position_basis)
        .raw_data
    )
    _assert_langevin_dynamics(psi, p @ p / 2 + shift, [collapse], p, rtol=1e-8)


def test_periodic_caldeira_leggett_langevin_dynamics() -> None:
    params = LangevinParameters(
        temperature=1, lambda_=FRICTION, mass=1, hbar=1, boltzmann=1
    )
    metadata = spaced_volume_metadata_from_stacked_delta_x((np.array([200.0]),), (512,))
    periodic_basis = _get_periodic_basis(metadata)
    out_basis = operator_basis(periodic_basis)
    n = periodic_basis.size

    def as_matrix(op: operator.Operator[Any, Any]) -> _Matrix:
        return op.with_basis(out_basis).raw_data.reshape(n, n)

    p = as_matrix(operator.build.p(metadata, axis=0, hbar=1))
    collapse = [
        as_matrix(op) for op in _build_environment_operators(periodic_basis, params)
    ]
    psi = (
        state.build_coherent(metadata, (100.0,), (0.7,), sigma_0=(1.5,))
        .with_basis(periodic_basis)
        .raw_data
    )
    # The periodic operators match the Caldeira-Leggett model up to O((k sigma)^2)
    _assert_langevin_dynamics(psi, p @ p / 2, collapse, p, rtol=1e-2)


def test_windowed_wiener_noise_matches_qutip() -> None:
    dt = 0.01
    shape = (1, 4)
    expected = Wiener(0.0, dt, np.random.default_rng(3), shape)  # type: ignore lib
    actual = _WindowedWiener(0.0, dt, np.random.default_rng(3), shape)  # type: ignore lib

    t = 0.0
    for n_steps in [7, 1, 13, 5, 100, 3]:
        np.testing.assert_array_equal(
            actual.dW(t, n_steps),
            expected.dW(t, n_steps),  # type: ignore lib
        )
        t += n_steps * dt
    # Only the noise of the last step is stored
    assert actual.noise.shape[0] == n_steps  # type: ignore lib


@pytest.mark.parametrize("method", ["euler", "rouchon"])
def test_windowed_wiener_ssesolve_matches_qutip(method: str) -> None:
    def solve() -> np.ndarray[Any, np.dtype[np.complexfloating]]:
        result = qutip.ssesolve(  # type: ignore lib
            H=qutip.sigmax(),  # type: ignore lib
            psi0=qutip.basis(2, 0),  # type: ignore lib
            tlist=np.linspace(0, 1, 11),
            sc_ops=[0.3 * qutip.sigmaz(), 0.2 * qutip.destroy(2)],  # type: ignore lib
            e_ops=[qutip.sigmax(), qutip.sigmaz()],  # type: ignore lib
            ntraj=2,
            seeds=[1, 2],
            heterodyne=True,
            options={"method": method, "dt": 1e-3, "keep_runs_results": True},
        )
        return np.array(result.runs_expect)  # type: ignore lib

    expected = solve()
    with windowed_wiener():
        actual = solve()
    np.testing.assert_array_equal(actual, expected)
