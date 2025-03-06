from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, overload, override

import numpy as np
from slate_core import (
    Array,
    FundamentalBasis,
    SimpleMetadata,
    TupleBasis,
    array,
    ctype,
    linalg,
)
from slate_core import basis as _basis
from slate_core.basis import Basis, BasisStateMetadata, TupleBasisLike
from slate_core.metadata import BasisMetadata

from slate_quantum.metadata import EigenvalueMetadata

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


class StateBuilder[B: Basis, DT: np.dtype[np.complexfloating]](
    array.ArrayBuilder[B, DT]
):
    @override
    def ok[DT_: np.complexfloating](
        self: StateBuilder[Basis[Any, ctype[DT_]], np.dtype[DT_]],
    ) -> State[B, DT]:
        return cast("Any", State(self._basis, self._data, 0))  # type: ignore safe to construct


class StateConversion[
    M0: BasisMetadata,
    B1: Basis,
    DT: np.dtype[np.complexfloating],
](array.ArrayConversion[M0, B1, DT]):
    @override
    def ok[M_: BasisMetadata, DT_: np.complexfloating](
        self: StateConversion[M_, Basis[M_, ctype[DT_]], np.dtype[DT_]],
    ) -> State[B1, DT]:
        return cast(
            "State[B1, DT]",
            State.build(
                self._new_basis,
                self._old_basis.__convert_vector_into__(
                    self._data, self._new_basis
                ).ok(),
            ).ok(),
        )


class State[
    B: Basis = Basis,
    DT: np.dtype[np.complexfloating] = np.dtype[np.complexfloating],
](Array[B, DT]):
    """represents a state vector in a basis."""

    @override
    def with_basis[
        DT_: np.dtype[np.complexfloating],
        M0_: BasisMetadata,
        B1_: Basis,
    ](
        self: State[Basis[M0_, Any], DT_],
        basis: B1_,
    ) -> StateConversion[M0_, B1_, DT_]:
        """Get the Array with the basis set to basis."""
        return StateConversion(self.raw_data, self.basis, basis)

    @override
    @staticmethod
    def build[B_: Basis, DT_: np.dtype[np.complexfloating]](
        basis: B_, data: np.ndarray[Any, DT_]
    ) -> StateBuilder[B_, DT_]:
        return StateBuilder(basis, data)


def inner_product[B: Basis](
    state_0: State[B],
    state_1: State[B],
) -> complex:
    """Calculate the inner product of two states."""
    product = linalg.einsum("i', i -> i", state_0, state_1)
    return np.sum(product.as_array()).item(0)


def normalization(
    state: State[Basis],
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


def normalize[M: BasisMetadata](
    state: State[Basis[M]],
) -> State[Basis[M]]:
    norm = normalization(state)
    state_as_index = array.as_mul_basis(state)

    return State.build(
        state_as_index.basis, state_as_index.raw_data / np.sqrt(norm)
    ).ok()


def get_occupations[B: Basis](
    state: State[B],
) -> Array[FundamentalBasis[BasisStateMetadata[B]], np.dtype[np.floating]]:
    state_cast = array.cast_basis(
        state, FundamentalBasis(BasisStateMetadata(state.basis))
    ).ok()
    return (
        array.abs(linalg.einsum("i', i -> i", state_cast, state_cast))
        .with_basis(state_cast.basis)
        .ok()
    )


TupleBasis2D = TupleBasisLike[tuple[BasisMetadata, BasisMetadata], None]


class StateListBuilder[B: TupleBasis2D, DT: np.dtype[np.complexfloating]](
    array.ArrayBuilder[B, DT]
):
    @override
    def ok[DT_: np.complexfloating](
        self: StateListBuilder[Basis[Any, ctype[DT_]], np.dtype[DT_]],
    ) -> StateList[B, DT]:
        return cast("Any", StateList(self._basis, self._data, 0))  # type: ignore safe to construct


class StateListConversion[
    M0: BasisMetadata,
    B1: TupleBasis2D,
    DT: np.dtype[np.complexfloating],
](array.ArrayConversion[M0, B1, DT]):
    @override
    def ok[M_: BasisMetadata, DT_: np.complexfloating](
        self: StateListConversion[M_, Basis[M_, ctype[DT_]], np.dtype[DT_]],
    ) -> StateList[B1, DT]:
        return cast(
            "StateList[B1, DT]",
            StateList.build(
                self._new_basis,
                self._old_basis.__convert_vector_into__(
                    self._data, self._new_basis
                ).ok(),
            ).ok(),
        )


class StateList[
    B: TupleBasis2D = TupleBasis2D,
    DT: np.dtype[np.complexfloating] = np.dtype[np.complexfloating],
](Array[B, DT]):
    """represents a state vector in a basis."""

    @override
    def with_basis[
        DT_: np.dtype[np.complexfloating],
        M0_: BasisMetadata,
        B1_: TupleBasis2D,
    ](
        self: Array[Basis[M0_, Any], DT_],
        basis: B1_,
    ) -> StateListConversion[M0_, B1_, DT_]:
        """Get the Array with the basis set to basis."""
        return StateListConversion(self.raw_data, self.basis, basis)

    @override
    def __iter__(self, /) -> Iterator[State[Basis, DT]]:  # type: ignore bad overload
        return (State.build(a.basis, a.raw_data).ok() for a in super().__iter__())

    @overload
    def __getitem__[M1_: BasisMetadata](
        self: StateList[Any, M1_], index: tuple[int, slice[None]]
    ) -> State[M1_]: ...

    @overload
    def __getitem__[M1_: BasisMetadata, I: slice | tuple[array.NestedIndex, ...]](
        self: StateList[Any, M1_], index: tuple[I, slice[None]]
    ) -> StateList[Any, M1_]: ...

    @overload
    def __getitem__[DT: np.generic](self: Array[Any, DT], index: int) -> DT: ...

    @overload
    def __getitem__[DT: np.generic](
        self: Array[Any, DT], index: tuple[array.NestedIndex, ...] | slice
    ) -> Array[Any, DT]: ...

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
        final_basis = TupleBasis((basis, _basis.as_tuple_basis(self.basis)[1]))
        return StateList(
            final_basis, self.basis.__convert_vector_into__(self.raw_data, final_basis)
        )

    @staticmethod
    def from_states[M_: BasisMetadata, DT_: np.dtype[np.complexfloating]](
        iter_: Iterable[State[Basis[M_], DT_]],
    ) -> StateList[TupleBasisLike[tuple[SimpleMetadata, M_], None], DT_]:
        states = list(iter_)
        assert all(x.basis == states[0].basis for x in states)

        list_basis = FundamentalBasis.from_size(len(states))
        state_basis = cast(
            "TupleBasis[tuple[FundamentalBasis, Basis[M_]], None, ctype[np.complexfloating]]",
            TupleBasis((list_basis, states[0].basis)),
        ).downcast_metadata()
        return StateList.build(
            state_basis,
            cast("np.ndarray[Any, DT_]", np.array([x.raw_data for x in states])),
        ).ok()

    @override
    @staticmethod
    def build[B_: TupleBasis2D, DT_: np.dtype[np.complexfloating]](
        basis: B_, data: np.ndarray[Any, DT_]
    ) -> StateListBuilder[B_, DT_]:
        return StateListBuilder(basis, data)


def all_inner_product[
    M0: BasisMetadata,
    M1: BasisMetadata,
    DT: np.dtype[np.complexfloating],
](
    state_0: StateList[TupleBasisLike[tuple[M0, M1], None], DT],
    state_1: StateList[TupleBasisLike[tuple[M0, M1], None], DT],
) -> Array[Basis[M0], DT]:
    """Calculate the inner product of two states."""
    return linalg.einsum("(j i'),(j i) ->j", state_0, state_1)


def normalize_all[
    M0: BasisMetadata,
    M1: BasisMetadata,
    DT: np.dtype[np.complexfloating],
](
    states: StateList[TupleBasisLike[tuple[M0, M1], None], DT],
) -> StateList[TupleBasisLike[tuple[M0, M1], None], DT]:
    norms = all_inner_product(states, states)
    norms = array.as_index_basis(norms)
    states = states.with_list_basis(norms.basis)
    states = states.with_state_basis(_basis.as_mul_basis(states.basis[1]))
    return StateList.build(
        states.basis,
        states.raw_data.reshape(states.basis.shape)
        / np.sqrt(norms.raw_data)[:, np.newaxis],
    ).ok()


@overload
def get_all_occupations[M0: BasisMetadata, B: Basis[BasisMetadata, Any]](
    states: StateList[M0, Any, TupleBasis2D[np.complexfloating, Any, B, None]],
) -> Array[
    TupleBasisLike[tuple[M0, BasisStateMetadata[B]], None, ctype[np.floating]],
    np.dtype[np.floating],
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
    states: StateList,
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
    basis = TupleBasis(
        (
            states_as_tuple.basis.children[0],
            FundamentalBasis(BasisStateMetadata(states_as_tuple.basis.children[1])),
        )
    )

    cast_states = array.cast_basis(states_as_tuple, basis)
    return (
        array.abs(linalg.einsum("(m i'),(m i) -> (m i)", cast_states, cast_states))
        .with_basis(cast_states.basis)
        .ok()
    )


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


type EigenstateList[M: BasisMetadata] = StateList[EigenvalueMetadata, M]
