from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.constants import hbar  # type: ignore lib
from slate_core import Array, Ctype, TupleBasis, TupleMetadata, array
from slate_core.basis import (
    AsUpcast,
    Basis,
    DiagonalBasis,
)
from slate_core.metadata import BasisMetadata

from slate_quantum.operator import into_diagonal_hermitian
from slate_quantum.state import State, StateList

try:
    import qutip  # type: ignore lib
    import qutip.ui  # type: ignore lib
except ImportError:
    qutip = None

if TYPE_CHECKING:
    from slate_core.basis import (
        TupleBasis2D,
    )

    from slate_quantum.metadata import TimeMetadata
    from slate_quantum.operator._operator import Operator, OperatorBasis


def _solve_schrodinger_equation_diagonal[
    B: Basis[BasisMetadata, Ctype[np.complexfloating]],
    TB: Basis[TimeMetadata, Ctype[np.complexfloating]],
](
    initial_state: State[Basis],
    times: TB,
    hamiltonian: Array[
        DiagonalBasis[TupleBasis[tuple[B, B], Any], Ctype[np.complexfloating]],
        np.dtype[np.number],
    ],
) -> StateList[
    TupleBasis2D[tuple[TB, B], None],
    np.dtype[np.complexfloating],
]:
    coefficients = (
        initial_state.with_basis(hamiltonian.basis.inner.children[0]).ok().raw_data
    )
    eigenvalues = hamiltonian.raw_data

    time_values = np.array(list(times.metadata().values))[times.points]
    vectors = coefficients[np.newaxis, :] * np.exp(
        -1j * eigenvalues * time_values[:, np.newaxis] / hbar
    )
    return StateList.build(
        TupleBasis((times, hamiltonian.basis.inner.children[0])).upcast(), vectors
    ).assert_ok()


def solve_schrodinger_equation_decomposition[
    M: BasisMetadata,
    TB: Basis[TimeMetadata, Ctype[np.complexfloating]],
](
    initial_state: State[Basis],
    times: TB,
    hamiltonian: Operator[OperatorBasis[M], np.dtype[np.complexfloating]],
) -> StateList[
    AsUpcast[
        TupleBasis[tuple[TB, Basis[M]], None],
        TupleMetadata[tuple[TimeMetadata, M], None],
    ],
    np.dtype[np.complexfloating],
]:
    """Solve the schrodinger equation by directly finding eigenstates for the given initial state and hamiltonian."""
    diagonal = into_diagonal_hermitian(hamiltonian)
    diagonal = array.cast_basis(diagonal, diagonal.basis.inner).assert_ok()
    return _solve_schrodinger_equation_diagonal(initial_state, times, diagonal)  # type: ignore cant infer M type


def solve_schrodinger_equation[
    M: BasisMetadata,
    TB: Basis[TimeMetadata, Ctype[np.complexfloating]],
](
    initial_state: State[Basis[M, Ctype[np.complexfloating]]],
    times: TB,
    hamiltonian: Operator[OperatorBasis[M], np.dtype[np.complexfloating]],
) -> StateList[TupleBasis2D[tuple[TB, Basis[M]]], np.dtype[np.complexfloating]]:
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

    hamiltonian_as_tuple = array.as_tuple_basis(hamiltonian)
    hamiltonian_data = hamiltonian_as_tuple.raw_data.reshape(
        hamiltonian_as_tuple.basis.shape
    )
    hamiltonian_qobj = qutip.Qobj(
        hamiltonian_data / hbar,
    )
    state_basis = cast(
        "Basis[M, Ctype[np.complexfloating]]", hamiltonian_as_tuple.basis.children[0]
    )
    initial_state_qobj = qutip.Qobj(initial_state.with_basis(state_basis).ok().raw_data)
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
    return StateList.build(
        TupleBasis((times, state_basis)).upcast(),
        np.array(
            np.asarray([state.full().reshape(-1) for state in result.states]),  # type: ignore lib
            dtype=np.complex128,
        ).reshape(-1),
    ).assert_ok()
