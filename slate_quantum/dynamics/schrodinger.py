from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.constants import hbar  # type: ignore lib
from slate_core import basis
from slate_core.metadata import BasisMetadata

from slate_quantum._util.legacy import LegacyBasis, LegacyDiagonalBasis, tuple_basis
from slate_quantum.operator import into_diagonal_hermitian
from slate_quantum.state._state import build_legacy_state_list

try:
    import qutip  # type: ignore lib
    import qutip.ui  # type: ignore lib
except ImportError:
    qutip = None

if TYPE_CHECKING:
    from slate_quantum._util.legacy import LegacyTupleBasis2D
    from slate_quantum.metadata import TimeMetadata
    from slate_quantum.operator._operator import LegacyOperator
    from slate_quantum.state import LegacyState, LegacyStateList
    from slate_quantum.state._basis import LegacyEigenstateBasis


def _solve_schrodinger_equation_diagonal[
    M: BasisMetadata,
    TB: LegacyBasis[TimeMetadata, np.complexfloating],
    B: LegacyBasis[BasisMetadata, np.complexfloating] = LegacyBasis[
        M, np.complexfloating
    ],
](
    initial_state: LegacyState[BasisMetadata],
    times: TB,
    hamiltonian: LegacyOperator[
        M,
        np.number[Any],
        LegacyDiagonalBasis[np.complexfloating, B, B, Any],
    ],
) -> LegacyStateList[
    TimeMetadata,
    M,
    LegacyTupleBasis2D[np.complexfloating, TB, B, None],
]:
    coefficients = initial_state.with_basis(hamiltonian.basis.inner[0]).raw_data
    eigenvalues = hamiltonian.raw_data

    time_values = np.array(list(times.metadata().values))[times.points]
    vectors = coefficients[np.newaxis, :] * np.exp(
        -1j * eigenvalues * time_values[:, np.newaxis] / hbar
    )
    return build_legacy_state_list(
        tuple_basis((times, hamiltonian.basis.inner[0])), vectors
    )


def solve_schrodinger_equation_decomposition[
    M: BasisMetadata,
    TB: LegacyBasis[TimeMetadata, np.complexfloating],
](
    initial_state: LegacyState[BasisMetadata],
    times: TB,
    hamiltonian: LegacyOperator[M, np.complexfloating],
) -> LegacyStateList[
    TimeMetadata,
    M,
    LegacyTupleBasis2D[np.complexfloating, TB, LegacyEigenstateBasis[M], None],
]:
    """Solve the schrodinger equation by directly finding eigenstates for the given initial state and hamiltonian."""
    diagonal = into_diagonal_hermitian(hamiltonian)
    return _solve_schrodinger_equation_diagonal(initial_state, times, diagonal)


def solve_schrodinger_equation[
    M: BasisMetadata,
    TB: LegacyBasis[TimeMetadata, np.complexfloating],
](
    initial_state: LegacyState[BasisMetadata],
    times: TB,
    hamiltonian: LegacyOperator[M, np.complexfloating],
) -> LegacyStateList[
    TimeMetadata,
    M,
    LegacyTupleBasis2D[
        np.complexfloating, TB, LegacyBasis[M, np.complexfloating], None
    ],
]:
    """Solve the schrodinger equation iteratively for the given initial state and hamiltonian.

    Internally, this function makes use of the qutip package.

    Raises
    ------
    ImportError
        If qutip is not installed
    """
    if qutip is None:
        msg = "The qutip package is required to use this function. Please install it with `pip install qutip`."
        raise ImportError(msg)

    hamiltonian_as_tuple = hamiltonian.with_basis(basis.as_tuple(hamiltonian.basis))
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
    return build_legacy_state_list(
        tuple_basis((times, hamiltonian_as_tuple.basis[0])),
        np.array(
            np.asarray([state.full().reshape(-1) for state in result.states]),  # type: ignore lib
            dtype=np.complex128,
        ).reshape(-1),
    )
