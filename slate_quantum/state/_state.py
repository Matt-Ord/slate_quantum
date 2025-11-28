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


class State[
    B: Basis = Basis,
    DT: np.dtype[np.generic] = np.dtype[np.complexfloating],
](Array[B, DT]):
    """represents a state vector in a basis."""

    @override
    def with_basis[B1_: Basis](self, basis: B1_) -> State[B1_, DT]:
        """Get the Array with the basis set to basis."""
        basis.ctype.assert_supports_dtype(self.dtype)
        assert basis.metadata() == self.basis.metadata()
        new_data = self.basis.__convert_vector_into__(self.raw_data, basis).ok()
        return State(basis, new_data)  # type: ignore[return-value]


type StateWithMetadata[M: BasisMetadata] = State[Basis[M], np.dtype[np.complexfloating]]


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
    """
    product = inner_product(state, state)
    return np.abs(product)


def normalize[M: BasisMetadata](
    state: State[Basis[M]],
) -> State[Basis[M]]:
    norm = normalization(state)
    state_as_index = array.as_mul_basis(state)

    return State(state_as_index.basis, state_as_index.raw_data / np.sqrt(norm))


def get_occupations[B: Basis](
    state: State[B],
) -> Array[FundamentalBasis[BasisStateMetadata[B]], np.dtype[np.floating]]:
    state_cast = array.cast_basis(
        state, FundamentalBasis(BasisStateMetadata(state.basis))
    )
    return array.abs(linalg.einsum("i', i -> i", state_cast, state_cast)).with_basis(
        state_cast.basis
    )


class StateList[
    B: TupleBasisLike2D = TupleBasisLike2D,
    DT: np.dtype[np.generic] = np.dtype[np.complexfloating],
](Array[B, DT]):
    """represents a state vector in a basis."""

    @overload
    def with_basis[B1_: TupleBasisLike2D](self, basis: B1_) -> StateList[B1_, DT]: ...

    @overload
    def with_basis[B1_: Basis](self, basis: B1_) -> Array[B1_, DT]: ...
    @override
    def with_basis[B1_: Basis](self, basis: B1_) -> Array[B1_, DT]:
        """Get the Array with the basis set to basis."""
        basis.ctype.assert_supports_dtype(self.dtype)
        assert basis.metadata() == self.basis.metadata()
        new_data = self.basis.__convert_vector_into__(self.raw_data, basis).ok()  # type: ignore[return-value]
        return StateList(basis, new_data)  # type: ignore[return-value]

    @override
    def __iter__[M1_: BasisMetadata](  # type: ignore bad overload
        self: StateList[TupleBasisLike2D[tuple[Any, M1_]]], /
    ) -> Iterator[State[Basis[M1_], DT]]:
        return (State(a.basis, a.raw_data) for a in super().__iter__())  # type: ignore bad overload

    @overload
    def __getitem__[M1_: BasisMetadata](
        self: StateList[TupleBasisLike2D[tuple[Any, M1_]]],
        index: tuple[int, slice[None]],
    ) -> State[Basis[M1_], DT]: ...

    @overload
    def __getitem__[
        M1_: BasisMetadata,
        I: slice | tuple[array.NestedIndex, ...],
        DT_: np.dtype[np.generic],
    ](
        self: StateList[TupleBasisLike2D[tuple[Any, M1_]], DT_],
        index: tuple[I, slice[None]],
    ) -> StateList[TupleBasisLike2D[tuple[Any, M1_]], DT_]: ...

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
        out = cast("Array[Basis[Any, Any], Any]", super()).__getitem__(index)
        if (
            isinstance(index, tuple)
            and isinstance(index[0], int)
            and index[1] == slice(None)
        ):
            return State(out.basis, out.raw_data)
        if isinstance(index, tuple) and index[1] == slice(None):
            return StateList(cast("Any", out.basis), out.raw_data)
        return out

    def with_state_basis[M: BasisMetadata, B1: Basis](  # B1: B
        self: StateList[TupleBasisLike2D[tuple[M, Any]], DT], basis: B1
    ) -> StateList[
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
    ) -> StateList[
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
        return StateList(
            state_basis,
            cast("np.ndarray[Any, DT_]", np.array([x.raw_data for x in states])),
        )


type StateListWithMetadata[M0: BasisMetadata, M1: BasisMetadata] = StateList[
    TupleBasisLike[tuple[M0, M1], None], np.dtype[np.complexfloating]
]


def inner_product_each[
    M0: BasisMetadata,
    M1: BasisMetadata,
    DT: np.dtype[np.complexfloating],
](
    state_0: StateList[TupleBasisLike[tuple[M0, M1], None], DT],
    state_1: StateList[TupleBasisLike[tuple[M0, M1], None], DT],
) -> Array[Basis[M0], DT]:
    """Calculate the inner product of two states."""
    return linalg.einsum("(j i'),(j i) ->j", state_0, state_1)


def all_inner_product[
    M0: BasisMetadata,
    M1: BasisMetadata,
    M2: BasisMetadata,
    DT: np.dtype[np.complexfloating],
](
    state_0: StateList[TupleBasisLike[tuple[M0, M2], None], DT],
    state_1: StateList[TupleBasisLike[tuple[M1, M2], None], DT],
) -> Array[TupleBasisLike[tuple[M0, M1], None], DT]:
    """Calculate the inner product of two states."""
    return linalg.einsum("(j i'),(k i) ->(j k)", state_0, state_1)


def normalize_all[
    M: TupleMetadata[tuple[BasisMetadata, BasisMetadata], None],
    DT: np.dtype[np.complexfloating],
](
    states: StateList[Basis[M], DT],
) -> StateList[Basis[M], DT]:
    norms = inner_product_each(states, states)
    norms = array.as_index_basis(norms)
    as_index = states.with_list_basis(norms.basis)
    as_mul = _basis.as_mul(as_index.basis.inner.children[1])
    states_as_mul = states.with_state_basis(as_mul)
    return StateList(
        cast("Basis[M]", states_as_mul.basis),
        cast(
            "np.ndarray[Any, DT]",
            states_as_mul.raw_data.reshape(states_as_mul.basis.inner.shape)
            / np.sqrt(norms.raw_data)[:, np.newaxis],
        ),
    )


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

    cast_states = array.cast_basis(states_as_tuple, basis)
    return array.abs(
        linalg.einsum("(m i'),(m i) -> (m i)", cast_states, cast_states)
    ).with_basis(cast_states.basis)


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
    occupations = array.cast_basis(occupations, average_basis)
    average = array.flatten(
        array.average(occupations, axis=0).with_basis(TupleBasis((state_basis,)))
    )
    std = array.flatten(
        array.standard_deviation(occupations, axis=0).with_basis(
            TupleBasis((state_basis,))
        )
    )
    std *= np.sqrt(1 / occupations.basis.shape[0])

    return average, std


type EigenstateList[
    M: BasisMetadata,
    CT: Ctype[Never] = Ctype[Never],
    DT: np.dtype[np.complexfloating] = np.dtype[np.complexfloating],
] = StateList[TupleBasisLike[tuple[EigenvalueMetadata, M], CT], DT]
