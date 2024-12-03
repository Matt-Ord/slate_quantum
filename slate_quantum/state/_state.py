from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload, override

import numpy as np
from slate import Array, FundamentalBasis, array, tuple_basis
from slate import basis as _basis
from slate.basis import (
    Basis,
    BasisStateMetadata,
    TupleBasis2D,
)
from slate.metadata import BasisMetadata, Metadata2D

from slate_quantum.metadata import EigenvalueMetadata

if TYPE_CHECKING:
    from collections.abc import Iterator


class State[M: BasisMetadata, B: Basis[Any, np.complex128] = Basis[M, np.complex128]](
    Array[M, np.complex128, B]
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
        state_1.with_basis(state_0.basis.dual_basis()).raw_data,
        state_0.raw_data,
        axes=(0, 0),
    ).item(0)


def get_occupations[B: Basis[BasisMetadata, Any]](
    state: State[Any, B],
) -> Array[BasisStateMetadata[B], np.float64, FundamentalBasis[BasisStateMetadata[B]]]:
    return Array(
        FundamentalBasis(BasisStateMetadata(state.basis)),
        np.abs(state.raw_data * state.with_basis(state.basis.dual_basis()).raw_data),
    )


class StateList[
    M0: BasisMetadata,
    M1: BasisMetadata,
    B: Basis[Metadata2D[BasisMetadata, BasisMetadata, None], np.complex128] = Basis[
        Metadata2D[M0, M1, None], np.complex128
    ],
](Array[Metadata2D[M0, M1, None], np.complex128, B]):
    """represents a state vector in a basis."""

    @override
    def with_basis[B1: Basis[Any, Any]](  # B1: B
        self, basis: B1
    ) -> StateList[M0, M1, B1]:
        """Get the Operator with the basis set to basis."""
        return StateList(
            basis, self.basis.__convert_vector_into__(self.raw_data, basis)
        )

    def __iter__(self, /) -> Iterator[State[M1, Basis[Any, np.complex128]]]:
        as_tuple = self.with_basis(_basis.as_tuple_basis(self.basis))
        return (
            State(as_tuple.basis[1], row)
            for row in as_tuple.raw_data.reshape(as_tuple.basis.shape)
        )

    def __getitem__(self, /, index: int) -> State[M1, Basis[Any, np.complex128]]:
        as_tuple = self.with_basis(_basis.as_tuple_basis(self.basis))
        return State(
            as_tuple.basis[1], as_tuple.raw_data.reshape(as_tuple.basis.shape)[index]
        )

    @overload
    def with_state_basis[B0: Basis[Any, Any], B1: Basis[Any, Any]](  # B1: B
        self: StateList[Any, Any, TupleBasis2D[Any, B0, Any, None]], basis: B1
    ) -> StateList[M0, M1, TupleBasis2D[Any, B0, B1, None]]: ...

    @overload
    def with_state_basis[B1: Basis[Any, Any]](  # B1: B
        self, basis: B1
    ) -> StateList[M0, M1, TupleBasis2D[Any, Any, B1, None]]: ...

    def with_state_basis(  # B1: B
        self, basis: Basis[BasisMetadata, Any]
    ) -> StateList[M0, M1, Any]:
        """Get the Operator with the state basis set to basis."""
        final_basis = tuple_basis((_basis.as_tuple_basis(self.basis)[0], basis))
        return StateList(
            final_basis, self.basis.__convert_vector_into__(self.raw_data, final_basis)
        )

    @overload
    def with_list_basis[B0: Basis[Any, Any], B1: Basis[Any, Any]](  # B1: B
        self: StateList[Any, Any, TupleBasis2D[Any, Any, B1, None]], basis: B0
    ) -> StateList[M0, M1, TupleBasis2D[Any, B0, B1, None]]: ...

    @overload
    def with_list_basis[B0: Basis[Any, Any]](  # B1: B
        self, basis: B0
    ) -> StateList[M0, M1, TupleBasis2D[Any, B0, Any, None]]: ...

    def with_list_basis(  # B1: B
        self, basis: Basis[Any, Any]
    ) -> StateList[M0, M1, Any]:
        """Get the Operator with the operator basis set to basis."""
        final_basis = tuple_basis((basis, _basis.as_tuple_basis(self.basis)[1]))
        return StateList(
            final_basis, self.basis.__convert_vector_into__(self.raw_data, final_basis)
        )


@overload
def get_all_occupations[M0: BasisMetadata, B: Basis[BasisMetadata, Any]](
    states: StateList[M0, Any, TupleBasis2D[np.complex128, Any, B, None]],
) -> Array[
    Metadata2D[M0, BasisStateMetadata[B], None],
    np.float64,
    TupleBasis2D[
        np.float64,
        Basis[M0, Any],
        FundamentalBasis[BasisStateMetadata[B]],
        None,
    ],
]: ...


@overload
def get_all_occupations[M0: BasisMetadata, M1: BasisMetadata](
    states: StateList[M0, M1],
) -> Array[
    Metadata2D[M0, BasisStateMetadata[Basis[M1, Any]], None],
    np.float64,
    TupleBasis2D[
        np.float64,
        Basis[M0, Any],
        FundamentalBasis[BasisStateMetadata[Basis[M1, Any]]],
        None,
    ],
]: ...


def get_all_occupations[M0: BasisMetadata, B: Basis[Any, Any]](
    states: StateList[M0, Any],
) -> Array[
    Metadata2D[M0, BasisStateMetadata[Basis[Any, Any]], None],
    np.float64,
    TupleBasis2D[
        np.float64,
        Basis[M0, Any],
        FundamentalBasis[BasisStateMetadata[Basis[Any, Any]]],
        None,
    ],
]:
    states_basis = _basis.as_tuple_basis(states.basis)
    states = states.with_list_basis(_basis.as_index_basis(states_basis[0]))
    dual_states = states.with_state_basis(states.basis[1].dual_basis())

    return Array(
        tuple_basis(
            (states.basis[0], FundamentalBasis(BasisStateMetadata(states.basis[1])))
        ),
        np.abs(states.raw_data * dual_states.raw_data),
    )


@overload
def get_average_occupations[B: Basis[BasisMetadata, Any]](
    states: StateList[Any, Any, TupleBasis2D[np.complex128, Any, B, None]],
) -> Array[
    BasisStateMetadata[B],
    np.float64,
    FundamentalBasis[BasisStateMetadata[B]],
]: ...


@overload
def get_average_occupations[M1: BasisMetadata](
    states: StateList[Any, M1],
) -> Array[
    BasisStateMetadata[Basis[M1, Any]],
    np.float64,
    FundamentalBasis[BasisStateMetadata[Basis[M1, Any]]],
]: ...


def get_average_occupations(
    states: StateList[Any, Any, Any],
) -> Array[
    BasisStateMetadata[Basis[Any, Any]],
    np.float64,
    FundamentalBasis[BasisStateMetadata[Basis[Any, Any]]],
]:
    occupations = get_all_occupations(states)
    # Dont include empty entries in average
    list_basis = _basis.as_state_list(_basis.as_index_basis(occupations.basis[0]))
    average_basis = tuple_basis((list_basis, occupations.basis[1]))
    occupations = array.cast_basis(occupations, average_basis)
    return array.flatten(array.average(occupations, axis=0))


type EigenstateList[M: BasisMetadata] = StateList[EigenvalueMetadata, M]
