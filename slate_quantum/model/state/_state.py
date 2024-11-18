from __future__ import annotations

from typing import Any, Iterator, override

import numpy as np
from slate.array import SlateArray
from slate.basis import Basis
from slate.basis.stacked import as_tuple_basis
from slate.metadata import BasisMetadata

from slate_quantum.model._label import EigenvalueMetadata


class State[M: BasisMetadata, B: Basis[Any, np.complex128] = Basis[M, np.complex128]](
    SlateArray[M, np.complex128, B]
):
    """represents a state vector in a basis."""

    @override
    def with_basis[B1: Basis[Any, Any]](  # B1: B
        self, basis: B1
    ) -> State[M, B1]:
        """Get the Operator with the basis set to basis."""
        return State(basis, self.basis.__convert_vector_into__(self.raw_data, basis))


def calculate_normalization(
    state: State[BasisMetadata],
) -> np.float64:
    """
    calculate the normalization of a state.

    This should always be 1

    Parameters
    ----------
    state: StateVector[Any] | StateDualVector[Any]

    Returns
    -------
    float
    """
    return np.sum(np.abs(state.raw_data) ** 2).astype(np.float64)


def calculate_inner_product[M: BasisMetadata](
    state_0: State[M],
    state_1: State[M],
) -> complex:
    """
    Calculate the inner product of two states.

    Parameters
    ----------
    state_0 : StateVector[_B0Inv]
    state_1 : StateDualVector[_B0Inv]

    Returns
    -------
    np.complex_
    """
    return np.tensordot(
        state_1.with_basis(state_0.basis.conjugate_basis()).raw_data,
        state_0.raw_data,
        axes=(0, 0),
    ).item(0)


class StateList[
    M: BasisMetadata,
    B: Basis[Any, np.complex128] = Basis[M, np.complex128],
](SlateArray[M, np.complex128, B]):
    """represents a state vector in a basis."""

    @override
    def with_basis[B1: Basis[Any, Any]](  # B1: B
        self, basis: B1
    ) -> StateList[M, B1]:
        """Get the Operator with the basis set to basis."""
        return StateList(
            basis, self.basis.__convert_vector_into__(self.raw_data, basis)
        )

    def __iter__(self, /) -> Iterator[State[M, Basis[Any, np.complex128]]]:
        as_tuple = self.with_basis(as_tuple_basis(self.basis))
        return (
            State(as_tuple.basis[1], row)
            for row in as_tuple.raw_data.reshape(as_tuple.basis.shape)
        )

    def __getitem__(self, /, index: int) -> State[M, Basis[Any, np.complex128]]:
        as_tuple = self.with_basis(as_tuple_basis(self.basis))
        return State(
            as_tuple.basis[1], as_tuple.raw_data.reshape(as_tuple.basis.shape)[index]
        )


type EigenstateList[B: Basis[Any, np.complex128]] = StateList[EigenvalueMetadata]
