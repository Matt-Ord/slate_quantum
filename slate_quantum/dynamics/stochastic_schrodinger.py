from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any, TypedDict, Unpack, cast

import numpy as np
from scipy.constants import hbar  # type: ignore lib
from slate.basis import (
    TruncatedBasis,
    TupleBasis2D,
    as_tuple_basis,
    tuple_basis,
)
from slate.metadata import BasisMetadata, Metadata2D
from slate.util import timed

from slate_quantum.state import State, StateList

try:
    import sse_solver_py
except ImportError:
    sse_solver_py = None

if TYPE_CHECKING:
    from slate.basis import Basis
    from sse_solver_py import BandedData, SSEMethod

    from slate_quantum.metadata import EigenvalueMetadata, TimeMetadata
    from slate_quantum.operator import OperatorList
    from slate_quantum.operator._operator import Operator


def _get_operator_diagonals(
    operator: list[list[complex]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
    operator_array = np.array(operator)
    return np.array(
        [
            np.concatenate(
                [
                    np.diag(operator_array, k=-i),
                    np.diag(operator_array, k=len(operator) - i),
                ]
            )
            for i in range(len(operator))
        ]
    )


def _get_banded_operator(operator: list[list[complex]], threshold: float) -> BandedData:
    diagonals = _get_operator_diagonals(operator)
    above_threshold = np.linalg.norm(diagonals, axis=1) > threshold

    diagonals_filtered = np.array(diagonals)[above_threshold]

    zero_imag = np.abs(np.imag(diagonals_filtered)) < threshold
    diagonals_filtered[zero_imag] = np.real(diagonals_filtered[zero_imag])
    zero_real = np.abs(np.real(diagonals_filtered)) < threshold
    diagonals_filtered[zero_real] = 1j * np.imag(diagonals_filtered[zero_real])

    diagonals_filtered = cast("list[list[complex]]", diagonals_filtered.tolist())

    offsets = cast("list[int]", np.arange(len(operator))[above_threshold].tolist())

    if sse_solver_py is None:
        msg = "sse_solver_py is not installed, please install it using `pip install slate_quantum[sse_solver_py]`"
        raise ImportError(msg)
    return sse_solver_py.BandedData(
        diagonals=diagonals_filtered,
        offsets=offsets,
        shape=(len(operator), len(operator[0])),
    )


def _get_banded_operators(
    operators: list[list[list[complex]]], threshold: float
) -> list[BandedData]:
    return [_get_banded_operator(o, threshold) for o in operators]


class SSEConfig(TypedDict, total=False):
    """Configuration for the stochastic schrodinger equation solver."""

    n_trajectories: int
    n_realizations: int
    method: SSEMethod
    r_threshold: float


@timed
def solve_stochastic_schrodinger_equation_banded[
    M: BasisMetadata,
    TB: TruncatedBasis[TimeMetadata, np.complex128],
](
    initial_state: State[M],
    times: TB,
    hamiltonian: Operator[M, np.complex128],
    noise: OperatorList[
        EigenvalueMetadata,
        M,
        np.complex128,
        TupleBasis2D[
            np.complex128,
            Basis[EigenvalueMetadata, np.complex128],
            Basis[Metadata2D[M, M, Any], np.complex128],
            Any,
        ],
    ],
    **kwargs: Unpack[SSEConfig],
) -> StateList[
    TimeMetadata,
    M,
    TupleBasis2D[np.complex128, TB, Basis[M, np.complex128], None],
]:
    """Given an initial state, use the stochastic schrodinger equation to solve the dynamics of the system."""
    # We get the best numerical performace if we set the norm of the largest collapse operators
    # to be one. This prevents us from accumulating large errors when multiplying state * dt * operator * conj_operator
    dt = times.metadata().delta / times.fundamental_size
    r_threshold = kwargs.get("r_threshold", 1e-8)

    hamiltonian_tuple = hamiltonian.with_basis(as_tuple_basis(hamiltonian.basis))
    operators_data = [
        o.with_basis(hamiltonian_tuple.basis).raw_data.reshape(
            hamiltonian_tuple.basis.shape
        )
        * (np.sqrt(e) / np.sqrt(2))
        for o, e in zip(noise, noise.basis[0].metadata().values[noise.basis[0].points])
    ]

    banded_collapse = _get_banded_operators(
        [[list(x * np.sqrt(dt / hbar)) for x in o] for o in operators_data],
        r_threshold,
    )

    banded_h = _get_banded_operator(
        [
            list(x * (dt / hbar))
            for x in hamiltonian_tuple.raw_data.reshape(hamiltonian_tuple.basis.shape)
        ],
        r_threshold,
    )
    initial_state_converted = initial_state.with_basis(hamiltonian_tuple.basis[0])
    ts = datetime.datetime.now(tz=datetime.UTC)

    if sse_solver_py is None:
        msg = "sse_solver_py is not installed, please install it using `pip install slate_quantum[sse_solver_py]`"
        raise ImportError(msg)

    data = sse_solver_py.solve_sse_banded(
        list(initial_state_converted.raw_data),
        banded_h,
        banded_collapse,
        sse_solver_py.SimulationConfig(
            n=times.size,
            step=times.truncation.step,
            dt=1,
            n_trajectories=kwargs.get("n_trajectories", 1),
            n_realizations=kwargs.get("n_realizations", 1),
            method=kwargs.get("method", "Euler"),
        ),
    )

    te = datetime.datetime.now(tz=datetime.UTC)
    print(f"solve rust banded took: {(te - ts).total_seconds()} sec")  # noqa: T201
    return StateList(tuple_basis((times, hamiltonian_tuple.basis[0])), np.array(data))
