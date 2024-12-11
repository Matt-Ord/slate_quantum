from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.constants import hbar  # type: ignore lib
from slate.basis import (
    DiagonalBasis,
    TupleBasis2D,
    as_tuple_basis,
    tuple_basis,
)
from slate.metadata import BasisMetadata

from slate_quantum.operator import into_diagonal_hermitian
from slate_quantum.state import State, StateList

try:
    import qutip  # type: ignore lib
    import qutip.ui  # type: ignore lib
except ImportError:
    qutip = None

if TYPE_CHECKING:
    from slate.basis import Basis

    from slate_quantum.metadata import TimeMetadata
    from slate_quantum.operator._operator import Operator
    from slate_quantum.state._basis import EigenstateBasis


def _solve_schrodinger_equation_diagonal[
    M: BasisMetadata,
    TB: Basis[TimeMetadata, np.complex128],
    B: Basis[BasisMetadata, np.complex128] = Basis[M, np.complex128],
](
    initial_state: State[BasisMetadata],
    times: TB,
    hamiltonian: Operator[
        M,
        np.number[Any],
        DiagonalBasis[np.complex128, B, B, Any],
    ],
) -> StateList[
    TimeMetadata,
    M,
    TupleBasis2D[np.complex128, TB, B, None],
]:
    coefficients = initial_state.with_basis(hamiltonian.basis.inner[0]).raw_data
    eigenvalues = hamiltonian.raw_data

    time_values = np.array(list(times.metadata().values))[times.points]
    vectors = (
        coefficients[np.newaxis, :]
        * np.exp(-1j * eigenvalues * time_values[:, np.newaxis] / hbar)
    ).astype(np.complex128)
    return StateList(tuple_basis((times, hamiltonian.basis.inner[0])), vectors)


def solve_schrodinger_equation_decomposition[
    M: BasisMetadata,
    TB: Basis[TimeMetadata, np.complex128],
](
    initial_state: State[BasisMetadata],
    times: TB,
    hamiltonian: Operator[M, np.complex128],
) -> StateList[
    TimeMetadata,
    M,
    TupleBasis2D[np.complex128, TB, EigenstateBasis[M], None],
]:
    """Solve the schrodinger equation by directly finding eigenstates for the given initial state and hamiltonian."""
    diagonal = into_diagonal_hermitian(hamiltonian)
    return _solve_schrodinger_equation_diagonal(initial_state, times, diagonal)


def solve_schrodinger_equation[
    M: BasisMetadata,
    TB: Basis[TimeMetadata, np.complex128],
](
    initial_state: State[BasisMetadata],
    times: TB,
    hamiltonian: Operator[M, np.complex128],
) -> StateList[
    TimeMetadata,
    M,
    TupleBasis2D[np.complex128, TB, Basis[M, np.complex128], None],
]:
    """Solve the schrodinger equation iteratively for the given initial state and hamiltonian.

    Internally, this function makes use of the qutip package.
    """
    if qutip is None:
        msg = "The qutip package is required to use this function. Please install it with `pip install qutip`."
        raise ImportError(msg)

    hamiltonian_as_tuple = hamiltonian.with_basis(as_tuple_basis(hamiltonian.basis))
    hamiltonian_data = hamiltonian_as_tuple.raw_data.reshape(
        hamiltonian_as_tuple.basis.shape
    )
    hamiltonian_qobj = qutip.Qobj(
        hamiltonian_data / hbar,
    )
    initial_state_qobj = qutip.Qobj(
        initial_state.with_basis(hamiltonian_as_tuple.basis[0]).raw_data
    )
    time_values = np.array(list(times.metadata().values))[times.points]
    result = qutip.sesolve(  # type: ignore lib
        hamiltonian_qobj,
        initial_state_qobj,
        time_values,
        e_ops=[],
        options={
            "progress_bar": "enhanced",
            "store_states": True,
        },
    )
    return StateList(
        tuple_basis((times, hamiltonian_as_tuple.basis[0])),
        np.array(
            np.asarray([state.full().reshape(-1) for state in result.states]),  # type: ignore lib
            dtype=np.complex128,
        ).reshape(-1),
    )
