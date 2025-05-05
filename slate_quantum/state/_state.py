from __future__ import annotations

from typing import TYPE_CHECKING, Any, Never, cast, overload, override

import numpy as np
from slate_core import (
    Array,
    Ctype,
    FundamentalBasis,
    SimpleMetadata,
    TupleBasis,
    TupleMetadata,
    array,
    linalg,
)
from slate_core import basis as _basis
from slate_core.basis import (
    AsUpcast,
    Basis,
    BasisStateMetadata,
    TupleBasis2D,
    TupleBasisLike,
    TupleBasisLike2D,
)
from slate_core.metadata import BasisMetadata

from slate_quantum.metadata import EigenvalueMetadata

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


class StateBuilder[B: Basis, DT: np.dtype[np.complexfloating]](
    array.ArrayBuilder[B, DT]
):
    @override
    def ok[DT_: np.complexfloating](
        self: StateBuilder[Basis[Any, Ctype[DT_]], np.dtype[DT_]],
    ) -> State[B, DT]:
        return cast("Any", State(self._basis, self._data, 0))  # type: ignore safe to construct

    @override
    def assert_ok(self) -> State[B, DT]:
        assert self._basis.ctype.supports_dtype(self._data.dtype)
        return self.ok()  # type: ignore safe to construct


class StateConversion[
    M0: BasisMetadata,
    B1: Basis,
    DT: np.dtype[np.complexfloating],
](array.ArrayConversion[M0, B1, DT]):
    @override
    def ok[M_: BasisMetadata, DT_: np.complexfloating](
        self: StateConversion[M_, Basis[M_, Ctype[DT_]], np.dtype[DT_]],
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

    @override
    def assert_ok(self) -> State[B1, DT]:
        assert self._new_basis.ctype.supports_dtype(self._data.dtype)
        return self.ok()  # type: ignore safe to construct


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


type LegacyState[M: BasisMetadata] = State[Basis[M], np.dtype[np.complexfloating]]


def build_legacy_state(
    basis: Basis, data: np.ndarray[Any, np.dtype[np.complexfloating]]
) -> LegacyState[Any]:
    """Build a legacy state."""
    return State.build(basis, data).assert_ok()


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


class StateListBuilder[B: TupleBasisLike2D, DT: np.dtype[np.complexfloating]](
    array.ArrayBuilder[B, DT]
):
    @override
    def ok[DT_: np.complexfloating](
        self: StateListBuilder[Basis[Any, Ctype[DT_]], np.dtype[DT_]],
    ) -> StateList[B, DT]:
        return cast("Any", StateList(self._basis, self._data, 0))  # type: ignore safe to construct

    @override
    def assert_ok(self) -> StateList[B, DT]:
        assert self._basis.ctype.supports_dtype(self._data.dtype)
        return self.ok()  # type: ignore safe to construct


class StateListConversion[
    M0: BasisMetadata,
    B1: TupleBasisLike2D,
    DT: np.dtype[np.complexfloating],
](array.ArrayConversion[M0, B1, DT]):
    @override
    def ok[
        M_: TupleMetadata[tuple[BasisMetadata, BasisMetadata]],
        DT_: np.complexfloating,
    ](
        self: StateListConversion[M_, Basis[M_, Ctype[DT_]], np.dtype[DT_]],
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

    @override
    def assert_ok(self) -> StateList[B1, DT]:
        assert self._new_basis.ctype.supports_dtype(self._data.dtype)
        return self.ok()  # type: ignore safe to construct


class StateList[
    B: TupleBasisLike2D = TupleBasisLike2D,
    DT: np.dtype[np.complexfloating] = np.dtype[np.complexfloating],
](Array[B, DT]):
    """represents a state vector in a basis."""

    @override
    def with_basis[
        DT_: np.dtype[np.complexfloating],
        M0_: BasisMetadata,
        B1_: TupleBasisLike2D,
    ](
        self: Array[Basis[M0_, Any], DT_],
        basis: B1_,
    ) -> StateListConversion[M0_, B1_, DT_]:
        """Get the Array with the basis set to basis."""
        return StateListConversion(self.raw_data, self.basis, basis)

    @override
    def __iter__[M1_: BasisMetadata](  # type: ignore bad overload
        self: StateList[TupleBasisLike2D[tuple[Any, M1_]]], /
    ) -> Iterator[State[Basis[M1_], DT]]:
        return (State.build(a.basis, a.raw_data).ok() for a in super().__iter__())  # type: ignore bad overload

    @overload
    def __getitem__[M1_: BasisMetadata](
        self: StateList[TupleBasisLike2D[tuple[Any, M1_]]],
        index: tuple[int, slice[None]],
    ) -> State[Basis[M1_], DT]: ...

    @overload
    def __getitem__[M1_: BasisMetadata, I: slice | tuple[array.NestedIndex, ...]](
        self: StateList[TupleBasisLike2D[tuple[Any, M1_]]], index: tuple[I, slice[None]]
    ) -> StateList[TupleBasisLike2D[tuple[Any, M1_]], DT]: ...

    @overload
    def __getitem__[DT1: Ctype[Never], DT_: np.dtype[np.generic]](
        self: Array[Any, DT_], index: int
    ) -> DT_: ...
    @overload
    def __getitem__[DT1: Ctype[Never], DT_: np.dtype[np.generic]](
        self: Array[Basis[Any, DT1], DT_], index: tuple[array.NestedIndex, ...] | slice
    ) -> Array[Basis[BasisMetadata, DT1], DT_]: ...

    @override
    def __getitem__(self, index: array.NestedIndex) -> Any:  # type: ignore overload bad
        out = cast("Array[Any, Any]", super()).__getitem__(index)
        if (
            isinstance(index, tuple)
            and isinstance(index[0], int)
            and index[1] == slice(None)
        ):
            out = cast(
                "Array[Basis[Any, Ctype[np.generic]], np.dtype[np.complexfloating]]",
                out,
            )
            return State.build(out.basis, out.raw_data).ok()
        if isinstance(index, tuple) and index[1] == slice(None):
            out = cast(
                "Array[Basis[Any, Ctype[np.generic]], np.dtype[np.complexfloating]]",
                out,
            )
            return StateList.build(out.basis, out.raw_data).ok()
        return out

    def with_state_basis[M: BasisMetadata, B1: Basis](  # B1: B
        self: StateList[TupleBasisLike2D[tuple[M, Any]], DT], basis: B1
    ) -> StateListConversion[
        Any,
        AsUpcast[
            TupleBasis[tuple[Basis[M], B1], None],
            TupleMetadata[tuple[M, BasisMetadata], None],
        ],
        DT,
    ]:
        """Get the Operator with the operator basis set to basis."""
        lhs_basis = _basis.as_tuple(self.basis).children[0]
        out_basis = TupleBasis((lhs_basis, basis)).upcast()
        return self.with_basis(out_basis)

    def with_list_basis[M: BasisMetadata, B1: Basis](  # B1: B
        self: StateList[TupleBasisLike2D[tuple[Any, M]], DT], basis: B1
    ) -> StateListConversion[
        Any,
        AsUpcast[
            TupleBasis[tuple[B1, Basis[M]], None],
            TupleMetadata[tuple[BasisMetadata, M], None],
        ],
        DT,
    ]:
        """Get the Operator with the operator basis set to basis."""
        rhs_basis = _basis.as_tuple(self.basis).children[1]
        out_basis = TupleBasis((basis, rhs_basis)).upcast()
        return self.with_basis(out_basis)

    @staticmethod
    def from_states[
        M_: BasisMetadata,
        DT1: Ctype[Never],
        DT_: np.dtype[np.complexfloating],
    ](
        iter_: Iterable[State[Basis[M_, DT1], DT_]],
    ) -> StateList[TupleBasisLike[tuple[SimpleMetadata, M_], None], DT_]:
        states = list(iter_)
        assert all(x.basis == states[0].basis for x in states)

        list_basis = FundamentalBasis.from_size(len(states))
        state_basis = TupleBasis((list_basis, states[0].basis)).resolve_ctype().upcast()
        return StateList.build(
            state_basis,
            cast("np.ndarray[Any, DT_]", np.array([x.raw_data for x in states])),
        ).ok()

    @override
    @staticmethod
    def build[B_: TupleBasisLike2D, DT_: np.dtype[np.complexfloating]](
        basis: B_, data: np.ndarray[Any, DT_]
    ) -> StateListBuilder[B_, DT_]:
        return StateListBuilder(basis, data)


type LegacyStateList[M0: BasisMetadata, M1: BasisMetadata, B: Basis = Basis] = (
    StateList[TupleBasisLike[tuple[M0, M1], None], np.dtype[np.complexfloating]]
)


def build_legacy_state_list(
    basis: Basis, data: np.ndarray[Any, np.dtype[np.complexfloating]]
) -> LegacyStateList[Any, Any, Any]:
    """Build a legacy state list."""
    return StateList.build(basis, data).assert_ok()  # type: ignore legacy


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
) -> StateList[TupleBasis2D[tuple[Basis[M0], Basis[M1]], None], DT]:
    norms = all_inner_product(states, states)
    norms = array.as_index_basis(norms)
    as_index = states.with_list_basis(norms.basis).assert_ok()
    as_mul = _basis.as_mul(as_index.basis.inner.children[1])
    states_as_mul = states.with_state_basis(as_mul).assert_ok()
    return StateList.build(
        states_as_mul.basis,
        cast(
            "np.ndarray[Any, DT]",
            states_as_mul.raw_data.reshape(states_as_mul.basis.inner.shape)
            / np.sqrt(norms.raw_data)[:, np.newaxis],
        ),
    ).assert_ok()


@overload
def get_all_occupations[M0: BasisMetadata, B: Basis[BasisMetadata, Any]](
    states: StateList[TupleBasis2D[tuple[Basis[M0], B], Any]],
) -> Array[
    TupleBasis[
        tuple[Basis[M0], FundamentalBasis[BasisStateMetadata[B]]],
        None,
    ],
    np.dtype[np.floating],
]: ...


@overload
def get_all_occupations[M0: BasisMetadata, M1: BasisMetadata](
    states: StateList[TupleBasisLike[tuple[M0, M1], Any]],
) -> Array[
    TupleBasis[
        tuple[Basis[M0], FundamentalBasis[BasisStateMetadata[Basis[M1]]]],
        None,
    ],
    np.dtype[np.floating],
]: ...


def get_all_occupations[M0: BasisMetadata, M1: BasisMetadata](
    states: StateList[TupleBasisLike[tuple[M0, M1], Any]],
) -> Array[
    TupleBasis[
        tuple[Basis[M0], FundamentalBasis[BasisStateMetadata[Basis[M1]]]],
        None,
    ],
    np.dtype[np.floating],
]:
    states_as_tuple = array.as_tuple_basis(states)
    basis = TupleBasis(
        (
            states_as_tuple.basis.children[0],
            FundamentalBasis(BasisStateMetadata(states_as_tuple.basis.children[1])),
        )
    )

    cast_states = array.cast_basis(states_as_tuple, basis).assert_ok()
    return (
        array.abs(linalg.einsum("(m i'),(m i) -> (m i)", cast_states, cast_states))
        .with_basis(cast_states.basis)
        .assert_ok()
    )


@overload
def get_average_occupations[B: Basis[BasisMetadata, Any]](
    states: StateList[TupleBasis2D[tuple[Any, B], Any]],
) -> tuple[
    Array[FundamentalBasis[BasisStateMetadata[B]], np.dtype[np.float64]],
    Array[FundamentalBasis[BasisStateMetadata[B]], np.dtype[np.float64]],
]: ...


@overload
def get_average_occupations[M1: BasisMetadata](
    states: StateList[TupleBasisLike[tuple[Any, M1]]],
) -> tuple[
    Array[FundamentalBasis[BasisStateMetadata[Basis[M1]]], np.dtype[np.float64]],
    Array[FundamentalBasis[BasisStateMetadata[Basis[M1]]], np.dtype[np.float64]],
]: ...


def get_average_occupations[M1: BasisMetadata](
    states: StateList[TupleBasisLike[tuple[Any, M1]]],
) -> tuple[
    Array[FundamentalBasis[BasisStateMetadata[Basis[M1]]], np.dtype[np.floating]],
    Array[FundamentalBasis[BasisStateMetadata[Basis[M1]]], np.dtype[np.floating]],
]:
    occupations = get_all_occupations(states)
    # Dont include empty entries in average
    list_basis = _basis.as_state_list(_basis.as_index(occupations.basis.children[0]))
    state_basis = occupations.basis.children[1]
    average_basis = TupleBasis((list_basis, state_basis))
    # TODO: this is wrong - must convert first  # noqa: FIX002
    occupations = array.cast_basis(occupations, average_basis).assert_ok()
    average = array.flatten(
        array.average(occupations, axis=0)
        .with_basis(TupleBasis((state_basis,)))
        .assert_ok()
    )
    std = array.flatten(
        array.standard_deviation(occupations, axis=0)
        .with_basis(TupleBasis((state_basis,)))
        .assert_ok()
    )
    std *= np.sqrt(1 / occupations.basis.shape[0])

    return average, std


type EigenstateList[
    M: BasisMetadata,
    CT: Ctype[Never] = Ctype[Never],
    DT: np.dtype[np.complexfloating] = np.dtype[np.complexfloating],
] = StateList[TupleBasisLike[tuple[EigenvalueMetadata, M], CT], DT]
