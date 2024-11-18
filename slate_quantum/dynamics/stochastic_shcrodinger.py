from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any, TypedDict, Unpack

import numpy as np
from scipy.constants import hbar  # type: ignore lib
from slate.basis import TruncatedBasis
from slate.basis.stacked import (
    VariadicTupleBasis,
    as_tuple_basis,
    tuple_basis,
)
from slate.metadata import BasisMetadata
from slate.util import timed

from slate_quantum.model.state import State, StateList

try:
    import sse_solver_py
except ImportError:
    sse_solver_py = None

if TYPE_CHECKING:
    from slate.basis import Basis
    from slate.metadata.stacked import StackedMetadata
    from sse_solver_py import BandedData, SSEMethod

    from slate_quantum.model import EigenvalueMetadata, TimeMetadata
    from slate_quantum.model.operator import OperatorList
    from slate_quantum.model.operator._operator import Operator


def _get_operator_diagonals(
    operator: list[list[complex]],
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
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

    diagonals_filtered = diagonals_filtered.tolist()

    offsets = np.arange(len(operator))[above_threshold].tolist()

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
    hamiltonian: Operator[StackedMetadata[M, Any], np.complex128],
    noise: OperatorList[
        StackedMetadata[BasisMetadata, Any],
        np.complex128,
        VariadicTupleBasis[
            np.complex128,
            Basis[EigenvalueMetadata, np.complex128],
            Basis[StackedMetadata[M, Any], np.complex128],
            Any,
        ],
    ],
    **kwargs: Unpack[SSEConfig],
) -> StateList[
    BasisMetadata, VariadicTupleBasis[np.complex128, TB, Basis[M, np.complex128], None]
]:
    """Given an initial state, use the stochastic schrodinger equation to solve the dynamics of the system."""
    hamiltonian = hamiltonian.with_basis(as_tuple_basis(hamiltonian.basis))
    operators_data = [
        e * o.with_basis(hamiltonian.basis).raw_data.reshape(hamiltonian.basis.shape)
        for o, e in zip(noise, noise.basis[0].metadata.values)
    ]
    operators_norm = [np.linalg.norm(o) for o in operators_data]

    # We get the best numerical performace if we set the norm of the largest collapse operators
    # to be one. This prevents us from accumulating large errors when multiplying state * dt * operator * conj_operator
    max_norm = np.max(operators_norm)
    dt = (times.metadata.delta * max_norm**2 / hbar).item()
    r_threshold = kwargs.get("r_threshold", 1e-8)

    banded_collapse = _get_banded_operators(
        [[list(x / max_norm) for x in o] for o in operators_data],
        r_threshold / dt,
    )

    banded_h = _get_banded_operator(
        [
            list(x / max_norm**2)
            for x in hamiltonian.raw_data.reshape(hamiltonian.basis.shape)
        ],
        r_threshold / dt,
    )
    initial_state_converted = initial_state.with_basis(hamiltonian.basis[0])
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
            dt=dt,
            n_trajectories=kwargs.get("n_trajectories", 1),
            n_realizations=kwargs.get("n_realizations", 1),
            method=kwargs.get("method", "Euler"),
        ),
    )

    te = datetime.datetime.now(tz=datetime.UTC)
    print(f"solve rust banded took: {(te - ts).total_seconds()} sec")  # noqa: T201

    return StateList(tuple_basis((times, hamiltonian.basis[0])), np.array(data))
