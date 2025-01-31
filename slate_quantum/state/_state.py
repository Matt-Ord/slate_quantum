from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, overload, override

import numpy as np
from slate import Array, FundamentalBasis, array, linalg, tuple_basis
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


class State[
    M: BasisMetadata,
    B: Basis[Any, np.complexfloating] = Basis[M, np.complexfloating],
](Array[M, np.complexfloating, B]):
    """represents a state vector in a basis."""

    def __init__[
        B1: Basis[BasisMetadata, Any],
    ](
        self: State[Any, B1],
        basis: B1,
        data: np.ndarray[Any, np.dtype[np.complexfloating]],
    ) -> None:
        super().__init__(cast("Any", basis), cast("Any", data))

    @override
    def with_basis[B1: Basis[Any, Any]](  # B1: B
        self, basis: B1
    ) -> State[M, B1]:
        """Get the Operator with the basis set to basis."""
        return State(basis, self.basis.__convert_vector_into__(self.raw_data, basis))


def inner_product[M: BasisMetadata](
    state_0: State[M],
    state_1: State[M],
) -> complex:
    """Calculate the inner product of two states."""
    product = linalg.einsum("i', i -> i", state_0, state_1)
    return np.sum(product.as_array()).item(0)


def normalization(
    state: State[BasisMetadata],
) -> np.float64:
    """
    Calculate the normalization of a state.

    This should always be 1

    Parameters
    ----------
    state: StateVector[Any] | StateDualVector[Any]

    Returns
    -------
    float
    """
    product = inner_product(state, state)
    return np.abs(product)


def get_occupations[B: Basis[BasisMetadata, Any]](
    state: State[Any, B],
) -> Array[BasisStateMetadata[B], np.floating, FundamentalBasis[BasisStateMetadata[B]]]:
    state_cast = array.cast_basis(
        state, FundamentalBasis(BasisStateMetadata(state.basis))
    )
    return array.abs(linalg.einsum("i', i -> i", state_cast, state_cast)).with_basis(
        state_cast.basis
    )


class StateList[
    M0: BasisMetadata,
    M1: BasisMetadata,
    B: Basis[
        Metadata2D[BasisMetadata, BasisMetadata, None], np.complexfloating
    ] = Basis[Metadata2D[M0, M1, None], np.complexfloating],
](Array[Metadata2D[M0, M1, None], np.complexfloating, B]):
    """represents a state vector in a basis."""

    def __init__[
        B1: Basis[Metadata2D[BasisMetadata, BasisMetadata, None], np.complexfloating],
    ](
        self: StateList[Any, Any, B1],
        basis: B1,
        data: np.ndarray[Any, np.dtype[np.complexfloating]],
    ) -> None:
        super().__init__(cast("Any", basis), cast("Any", data))

    @override
    def with_basis[B1: Basis[Any, Any]](  # B1: B
        self, basis: B1
    ) -> StateList[M0, M1, B1]:
        """Get the Operator with the basis set to basis."""
        return StateList(
            basis, self.basis.__convert_vector_into__(self.raw_data, basis)
        )

    @override
    def __iter__(self, /) -> Iterator[State[M1, Basis[Any, np.complexfloating]]]:  # type: ignore bad overload
        return (State(a.basis, a.raw_data) for a in super().__iter__())

    @overload
    def __getitem__[_M1: BasisMetadata](
        self: StateList[Any, _M1], index: tuple[int, slice[None]]
    ) -> State[_M1]: ...

    @overload
    def __getitem__[_M1: BasisMetadata, _I: slice | tuple[array.NestedIndex, ...]](
        self: StateList[Any, _M1], index: tuple[_I, slice[None]]
    ) -> StateList[Any, _M1]: ...

    @overload
    def __getitem__[_DT: np.generic](self: Array[Any, _DT], index: int) -> _DT: ...

    @overload
    def __getitem__[_DT: np.generic](
        self: Array[Any, _DT], index: tuple[array.NestedIndex, ...] | slice
    ) -> Array[Any, _DT]: ...

    @override
    def __getitem__(self, index: array.NestedIndex) -> Any:  # type: ignore overload bad
        out = cast("Array[Any, Any]", super()).__getitem__(index)
        if (
            isinstance(index, tuple)
            and isinstance(index[0], int)
            and index[1] == slice(None)
        ):
            out = cast("Array[Any, Any]", out)
            return State(out.basis, out.raw_data)
        if isinstance(index, tuple) and index[1] == slice(None):
            return StateList(out.basis, out.raw_data)
        return out

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


def normalize_states[M0: BasisMetadata, M1: BasisMetadata](
    states: StateList[M0, M1],
) -> StateList[
    M0,
    M1,
    TupleBasis2D[np.complexfloating, Basis[M0, Any], Basis[M1, Any], None],
]:
    norms = all_inner_product(states, states)
    norms = array.as_index_basis(norms)
    states = states.with_list_basis(norms.basis)
    states = states.with_state_basis(_basis.as_mul_basis(states.basis[1]))
    return StateList(
        states.basis,
        states.raw_data.reshape(states.basis.shape)
        / np.sqrt(norms.raw_data)[:, np.newaxis],
    )


@overload
def get_all_occupations[M0: BasisMetadata, B: Basis[BasisMetadata, Any]](
    states: StateList[M0, Any, TupleBasis2D[np.complexfloating, Any, B, None]],
) -> Array[
    Metadata2D[M0, BasisStateMetadata[B], None],
    np.floating,
    TupleBasis2D[
        np.floating,
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
    np.floating,
    TupleBasis2D[
        np.floating,
        Basis[M0, Any],
        FundamentalBasis[BasisStateMetadata[Basis[M1, Any]]],
        None,
    ],
]: ...


def get_all_occupations[M0: BasisMetadata, B: Basis[Any, Any]](
    states: StateList[M0, Any],
) -> Array[
    Metadata2D[M0, BasisStateMetadata[Basis[Any, Any]], None],
    np.floating,
    TupleBasis2D[
        np.floating,
        Basis[M0, Any],
        FundamentalBasis[BasisStateMetadata[Basis[Any, Any]]],
        None,
    ],
]:
    states_as_tuple = array.as_tuple_basis(states)
    basis = tuple_basis(
        (
            states_as_tuple.basis[0],
            FundamentalBasis(BasisStateMetadata(states_as_tuple.basis[1])),
        )
    )

    cast_states = array.cast_basis(states_as_tuple, basis)
    return array.abs(
        linalg.einsum("(m i'),(m i) -> (m i)", cast_states, cast_states)
    ).with_basis(cast_states.basis)


@overload
def get_average_occupations[B: Basis[BasisMetadata, Any]](
    states: StateList[Any, Any, TupleBasis2D[np.complexfloating, Any, B, None]],
) -> tuple[
    Array[
        BasisStateMetadata[B],
        np.float64,
        FundamentalBasis[BasisStateMetadata[B]],
    ],
    Array[
        BasisStateMetadata[B],
        np.float64,
        FundamentalBasis[BasisStateMetadata[B]],
    ],
]: ...


@overload
def get_average_occupations[M1: BasisMetadata](
    states: StateList[Any, M1],
) -> tuple[
    Array[
        BasisStateMetadata[Basis[M1, Any]],
        np.float64,
        FundamentalBasis[BasisStateMetadata[Basis[M1, Any]]],
    ],
    Array[
        BasisStateMetadata[Basis[M1, Any]],
        np.float64,
        FundamentalBasis[BasisStateMetadata[Basis[M1, Any]]],
    ],
]: ...


def get_average_occupations(
    states: StateList[Any, Any, Any],
) -> tuple[
    Array[
        BasisStateMetadata[Basis[Any, Any]],
        np.floating,
        FundamentalBasis[BasisStateMetadata[Basis[Any, Any]]],
    ],
    Array[
        BasisStateMetadata[Basis[Any, Any]],
        np.floating,
        FundamentalBasis[BasisStateMetadata[Basis[Any, Any]]],
    ],
]:
    occupations = get_all_occupations(states)
    # Dont include empty entries in average
    list_basis = _basis.as_state_list(_basis.as_index_basis(occupations.basis[0]))
    average_basis = tuple_basis((list_basis, occupations.basis[1]))
    # TODO: this is wrong - must convert first  # noqa: FIX002
    occupations = array.cast_basis(occupations, average_basis)

    average = array.flatten(array.average(occupations, axis=0))
    std = array.flatten(array.standard_deviation(occupations, axis=0))
    std *= np.sqrt(1 / occupations.basis.shape[0])

    return average, std


def all_inner_product[M: BasisMetadata, M1: BasisMetadata](
    state_0: StateList[M, M1],
    state_1: StateList[M, M1],
) -> Array[M, np.complexfloating]:
    """Calculate the inner product of two states."""
    return linalg.einsum("(j i'),(j i) ->j", state_0, state_1)


type EigenstateList[M: BasisMetadata] = StateList[EigenvalueMetadata, M]
