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
from slate_core.basis import (
    AsUpcast,
    Basis,
    TupleBasis2D,
    TupleBasisLike,
    TupleBasisLike2D,
    are_dual_shapes,
    as_tuple_basis,
)
from slate_core.linalg import into_diagonal
from slate_core.metadata import BasisMetadata, NestedLength

from slate_quantum.state._state import State, StateList

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


def _assert_operator_basis(basis: Basis[BasisMetadata, Any]) -> None:
    is_dual = basis.is_dual
    if isinstance(is_dual, bool) or len(is_dual) != 2:  # noqa: PLR2004
        msg = "Basis is not 2d"
        raise TypeError(msg)
    assert are_dual_shapes(is_dual[0], is_dual[1])


type OperatorMetadata[M: BasisMetadata = BasisMetadata] = TupleMetadata[
    tuple[M, M], None
]
type OperatorBasis[
    M: BasisMetadata = BasisMetadata,
    CT: Ctype[Never] = Ctype[Never],
] = Basis[OperatorMetadata[M], CT]


def operator_basis[M: BasisMetadata, CT: Ctype[np.generic]](
    basis: Basis[M, CT],
) -> TupleBasisLike2D[tuple[M, M], None, CT]:
    return TupleBasis((basis, basis.dual_basis())).resolve_ctype().upcast()


class OperatorBuilder[B: OperatorBasis, DT: np.dtype[np.generic]](
    array.ArrayBuilder[B, DT]
):
    @override
    def ok[DT_: np.generic](
        self: OperatorBuilder[Basis[Any, Ctype[DT_]], np.dtype[DT_]],
    ) -> Operator[B, DT]:
        _assert_operator_basis(self._basis)
        return cast("Any", Operator(self._basis, self._data, 0))  # type: ignore safe to construct

    @override
    def assert_ok(self) -> Operator[B, DT]:
        assert self._basis.ctype.supports_dtype(self._data.dtype)
        return self.ok()  # type: ignore safe to construct


class OperatorConversion[
    M0: OperatorMetadata,
    B1: OperatorBasis,
    DT: np.dtype[np.generic],
](array.ArrayConversion[M0, B1, DT]):
    @override
    def ok[M_: OperatorMetadata, DT_: np.generic](
        self: OperatorConversion[M_, Basis[M_, Ctype[DT_]], np.dtype[DT_]],
    ) -> Operator[B1, DT]:
        return cast(
            "Operator[B1, DT]",
            Operator.build(
                self._new_basis,
                self._old_basis.__convert_vector_into__(
                    self._data, self._new_basis
                ).ok(),
            ).ok(),
        )

    @override
    def assert_ok(self) -> Operator[B1, DT]:
        assert self._new_basis.ctype.supports_dtype(self._data.dtype)
        return self.ok()  # type: ignore safe to construct


class Operator[
    B: OperatorBasis,
    DT: np.dtype[np.generic],
](Array[B, DT]):
    """Represents an operator in a quantum system."""

    @override
    @staticmethod
    def build[B_: OperatorBasis, DT_: np.dtype[np.generic]](
        basis: B_, data: np.ndarray[Any, DT_]
    ) -> OperatorBuilder[B_, DT_]:
        return OperatorBuilder(basis, data)

    @override
    def with_basis[
        DT_: np.dtype[np.generic],
        M0_: OperatorMetadata,
        B1_: OperatorBasis,
    ](
        self: Operator[Basis[M0_, Any], DT_],
        basis: B1_,
    ) -> OperatorConversion[M0_, B1_, DT_]:
        return OperatorConversion(self.raw_data, self.basis, basis)

    @overload
    def __add__[M1: OperatorMetadata, DT_: np.number](
        self: Operator[Basis[M1, Ctype[DT_]], np.dtype[DT_]],
        other: Operator[Basis[M1, Ctype[DT_]], np.dtype[DT_]],
    ) -> Operator[Basis[M1, Ctype[DT_]], np.dtype[DT_]]: ...
    @overload
    def __add__[M_: BasisMetadata, DT_: np.number](
        self: Array[Basis[M_, Ctype[DT_]], np.dtype[DT_]],
        other: Array[Basis[M_, Ctype[DT_]], np.dtype[DT_]],
    ) -> Array[Basis[M_, Ctype[DT_]], np.dtype[DT_]]: ...
    @override
    def __add__[M_: BasisMetadata, DT_: np.number](
        self: Array[Basis[M_, Ctype[DT_]], np.dtype[DT_]],
        other: Array[Basis[M_, Ctype[DT_]], np.dtype[DT_]],
    ) -> Array[Basis[M_, Ctype[DT_]], np.dtype[DT_]]:
        array = cast("Array[Any, np.dtype[DT_]]", super()).__add__(other)
        if isinstance(other, Operator):
            return cast("Any", Operator.build(array.basis, array.raw_data).assert_ok())
        return array

    @overload
    def __sub__[M1: OperatorMetadata, DT_: np.number](
        self: Operator[Basis[M1, Ctype[DT_]], np.dtype[DT_]],
        other: Operator[Basis[M1, Ctype[DT_]], np.dtype[DT_]],
    ) -> Operator[Basis[M1, Ctype[DT_]], np.dtype[DT_]]: ...
    @overload
    def __sub__[M_: BasisMetadata, DT_: np.number](
        self: Array[Basis[M_, Ctype[DT_]], np.dtype[DT_]],
        other: Array[Basis[M_, Ctype[DT_]], np.dtype[DT_]],
    ) -> Array[Basis[M_, Ctype[DT_]], np.dtype[DT_]]: ...
    @override
    def __sub__[M_: BasisMetadata, DT_: np.number](
        self: Array[Basis[M_, Ctype[DT_]], np.dtype[DT_]],
        other: Array[Basis[M_, Ctype[DT_]], np.dtype[DT_]],
    ) -> Array[Basis[M_, Ctype[DT_]], np.dtype[DT_]]:
        array = cast("Array[Any, np.dtype[DT_]]", super()).__sub__(other)
        if isinstance(other, Operator):
            return cast("Any", Operator.build(array.basis, array.raw_data).assert_ok())

        return array

    @override
    def __mul__[M_: OperatorMetadata, DT_: np.number](
        self: Operator[Basis[M_, Ctype[DT_]], np.dtype[DT_]],
        other: float,
    ) -> Operator[Basis[M_, Ctype[DT_]], np.dtype[DT_]]:
        # TODO: always support complex numbers  # noqa: FIX002
        out = cast("Array[Any, np.dtype[DT_]]", super()).__mul__(other)
        return Operator.build(out.basis, out.raw_data).assert_ok()

    def as_diagonal[M_: BasisMetadata, CT: Ctype[Never]](
        self: Operator[OperatorBasis[M_, CT], DT],
    ) -> Array[Basis[M_, CT], DT]:
        diagonal = into_diagonal(self)
        inner = cast("Basis[M, DT]", diagonal.basis.inner[1])
        return Array(inner, diagonal.raw_data)


def _assert_operator_list_basis(basis: Basis) -> None:
    is_dual = basis.is_dual
    if isinstance(is_dual, bool):
        msg = "Basis is not 2d"
        raise TypeError(msg)

    _assert_operator_basis(as_tuple_basis(basis).children[1])


def expectation[M: BasisMetadata](
    operator: Operator[OperatorBasis[M], np.dtype[np.complexfloating]],
    state: State[Basis[M]],
) -> complex:
    """Calculate the expectation value of an operator."""
    return (
        linalg.einsum(
            "i' ,(i j'),j -> ",
            array.dual_basis(state),
            operator,
            state,
        )
        .as_array()
        .item()
    )


def expectation_of_each[M0: BasisMetadata, M: BasisMetadata](
    operator: Operator[Basis[OperatorMetadata[M]], np.dtype[np.complexfloating]],
    states: StateList[TupleBasisLike[tuple[M0, M], None]],
) -> Array[Basis[M0], np.dtype[np.complexfloating]]:
    """Calculate the expectation value of an operator."""
    basis = as_tuple_basis(states.basis).children[1]
    return linalg.einsum(
        "(a i'),(i j'),(a j) -> a",
        states.with_state_basis(basis.dual_basis()).assert_ok(),
        operator,
        states,
    )


def apply[M: BasisMetadata](
    operator: Operator[Basis[OperatorMetadata[M]], np.dtype[np.complexfloating]],
    state: State[Basis[M]],
) -> State[Basis[M]]:
    """Apply an operator to a state."""
    out = linalg.einsum("(i j'),j -> i", operator, state)
    return State.build(out.basis, out.raw_data).ok()


def apply_to_each[M0: BasisMetadata, M: BasisMetadata](
    operator: Operator[Basis[OperatorMetadata[M]], np.dtype[np.complexfloating]],
    states: StateList[TupleBasisLike[tuple[M0, M], None]],
) -> StateList[TupleBasisLike[tuple[M0, M], None]]:
    """Apply an operator to a state."""
    out = linalg.einsum("(i j'),(k j) -> (k i)", operator, states)
    return StateList.build(out.basis, out.raw_data).ok()


type OperatorListMetadata[
    M0: BasisMetadata = BasisMetadata,
    M1: OperatorMetadata = OperatorMetadata,
] = TupleMetadata[tuple[M0, M1], Any]
type OperatorListBasis[
    M0: BasisMetadata = BasisMetadata,
    M1: OperatorMetadata = OperatorMetadata,
] = Basis[OperatorListMetadata[M0, M1]]


class OperatorListBuilder[
    B: OperatorListBasis,
    DT: np.dtype[np.generic],
](array.ArrayBuilder[B, DT]):
    @override
    def ok[DT_: np.generic](
        self: OperatorListBuilder[Basis[Any, Ctype[DT_]], np.dtype[DT_]],
    ) -> OperatorList[B, DT]:
        _assert_operator_list_basis(self._basis)
        return cast("Any", OperatorList(self._basis, self._data, 0))  # type: ignore safe to construct

    @override
    def assert_ok(self) -> OperatorList[B, DT]:
        assert self._basis.ctype.supports_dtype(self._data.dtype)
        return self.ok()  # type: ignore safe to construct


class OperatorListConversion[
    M0: OperatorListMetadata,
    B1: OperatorListBasis,
    DT: np.dtype[np.generic],
](array.ArrayConversion[M0, B1, DT]):
    @override
    def ok[M_: OperatorListMetadata, DT_: np.generic](
        self: OperatorListConversion[M_, Basis[M_, Ctype[DT_]], np.dtype[DT_]],
    ) -> OperatorList[B1, DT]:
        return cast(
            "OperatorList[B1, DT]",
            OperatorList.build(
                self._new_basis,
                self._old_basis.__convert_vector_into__(
                    self._data, self._new_basis
                ).ok(),
            ).ok(),
        )

    @override
    def assert_ok(self) -> OperatorList[B1, DT]:
        assert self._new_basis.ctype.supports_dtype(self._data.dtype)
        return self.ok()  # type: ignore safe to construct


class OperatorList[
    B: OperatorListBasis = OperatorListBasis,
    DT: np.dtype[np.generic] = np.dtype[np.generic],
](Array[B, DT]):
    """Represents an operator in a quantum system."""

    @property
    @override
    def fundamental_shape(self) -> tuple[NestedLength, NestedLength]:
        return cast("tuple[NestedLength, NestedLength]", super().fundamental_shape)

    @override
    @staticmethod
    def build[B_: OperatorListBasis, DT_: np.dtype[np.generic]](
        basis: B_, data: np.ndarray[Any, DT_]
    ) -> OperatorListBuilder[B_, DT_]:
        return OperatorListBuilder(basis, data)

    @override
    def with_basis[
        DT_: np.dtype[np.generic],
        M0_: OperatorListMetadata,
        B1_: OperatorListBasis,
    ](
        self: OperatorList[Basis[M0_, Any], DT_],
        basis: B1_,
    ) -> OperatorListConversion[M0_, B1_, DT_]:
        return OperatorListConversion(self.raw_data, self.basis, basis)

    def with_operator_basis[M: BasisMetadata, B1: OperatorBasis](  # B1: B
        self: OperatorList[OperatorListBasis[M, Any], DT], basis: B1
    ) -> OperatorListConversion[
        Any,
        AsUpcast[
            TupleBasis[tuple[Basis[M], B1], None],
            TupleMetadata[tuple[BasisMetadata, OperatorMetadata], None],
        ],
        DT,
    ]:
        """Get the Operator with the operator basis set to basis."""
        lhs_basis = as_tuple_basis(self.basis).children[0]
        out_basis = TupleBasis((lhs_basis, basis)).upcast()
        return self.with_basis(out_basis)

    def with_list_basis[M: OperatorMetadata, B1: Basis](  # B1: B
        self: OperatorList[OperatorListBasis[Any, M], DT], basis: B1
    ) -> OperatorListConversion[
        Any,
        AsUpcast[
            TupleBasis[tuple[B1, Basis[M]], None],
            TupleMetadata[tuple[BasisMetadata, OperatorMetadata], None],
        ],
        DT,
    ]:
        """Get the Operator with the operator basis set to basis."""
        rhs_basis = as_tuple_basis(self.basis).children[1]
        out_basis = TupleBasis((basis, rhs_basis)).upcast()
        return self.with_basis(out_basis)

    @overload
    def __iter__[M1_: BasisMetadata, B_: Basis[Any, Any], DT_: np.generic](
        self: OperatorList[Any, M1_, DT_, TupleBasis2D[Any, Any, B_, None]], /
    ) -> Iterator[Operator[M1_, DT_, B_]]: ...

    @overload
    def __iter__(self, /) -> Iterator[Operator[M1, Any]]: ...

    @override
    def __iter__(self, /) -> Iterator[Operator[M1, Any]]:  # type: ignore bad overload
        return (
            Operator[M1, DT](
                cast("Basis[Metadata2D[M1, M1, None], Any]", row.basis),
                cast("Any", row.raw_data),
            )
            for row in super().__iter__()
        )

    @overload
    def __getitem__[
        M1_: BasisMetadata,
        DT1: np.generic,
        B1: Basis[OperatorMetadata, Any] = Basis[Metadata2D[M1_, M1_, None], DT1],
    ](
        self: OperatorList[Any, M1_, DT1, TupleBasis2D[Any, Any, B1, None]],
        /,
        index: tuple[int, slice[None]],
    ) -> Operator[M1_, DT1, B1]: ...

    @overload
    def __getitem__[
        M1_: BasisMetadata,
        DT1: np.generic,
    ](
        self: OperatorList[Any, M1_, DT1, Any],
        /,
        index: tuple[int, slice[None, None, None]],
    ) -> Operator[M1_, DT1]: ...

    @overload
    def __getitem__[
        M1_: BasisMetadata,
        DT1: np.generic,
        I: slice | tuple[NestedIndex, ...],
    ](
        self: OperatorList[Any, M1_, DT1], index: tuple[I, slice[None]]
    ) -> OperatorList[Any, M1_, DT1]: ...

    @overload
    def __getitem__[DT_: np.generic](self: Array[Any, DT_], index: int) -> DT_: ...

    @overload
    def __getitem__[DT_: np.generic](
        self: Array[Any, DT_], index: tuple[NestedIndex, ...] | slice
    ) -> Array[Any, DT_]: ...

    @override
    def __getitem__[M1_: BasisMetadata, DT_: np.generic](  # type: ignore override
        self: Array[Any, DT_], index: NestedIndex
    ) -> Array[Any, DT_] | DT_ | Operator[Any, DT_]:
        out = cast("Array[Any, DT_]", super()).__getitem__(index)
        out = cast("Array[Any, DT_]", out)
        if (
            isinstance(index, tuple)
            and isinstance(index[0], int)
            and index[1] == slice(None)
        ):
            return Operator(out.basis, out.raw_data)

        if isinstance(index, tuple) and index[1] == slice(None):
            return OperatorList(out.basis, out.raw_data)
        return out

    @staticmethod
    def from_operators[M_: OperatorMetadata, DT_: np.dtype[np.generic]](
        iter_: Iterable[Operator[Basis[M_], DT_]],
    ) -> OperatorList[
        AsUpcast[
            TupleBasis[tuple[Basis[SimpleMetadata], Basis[M_]], None],
            OperatorListMetadata,
        ],
        DT_,
    ]:
        states = list(iter_)
        assert all(x.basis == states[0].basis for x in states)

        list_basis = FundamentalBasis.from_size(len(states))
        state_basis = cast(
            "TupleBasis[tuple[FundamentalBasis, Basis[M_]], None, Ctype[np.generic]]",
            TupleBasis((list_basis, states[0].basis)),
        ).upcast()
        return OperatorList.build(
            state_basis,
            cast("np.ndarray[Any, DT_]", np.array([x.raw_data for x in states])),
        ).ok()

    @overload
    def __add__[M1: OperatorListMetadata, DT_: np.number](
        self: OperatorList[Basis[M1, Ctype[DT_]], np.dtype[DT_]],
        other: OperatorList[Basis[M1, Ctype[DT_]], np.dtype[DT_]],
    ) -> OperatorList[Basis[M1, Ctype[DT_]], np.dtype[DT_]]: ...
    @overload
    def __add__[M_: BasisMetadata, DT_: np.number](
        self: Array[Basis[M_, Ctype[DT_]], np.dtype[DT_]],
        other: Array[Basis[M_, Ctype[DT_]], np.dtype[DT_]],
    ) -> Array[Basis[M_, Ctype[DT_]], np.dtype[DT_]]: ...
    @override
    def __add__[M_: BasisMetadata, DT_: np.number](
        self: Array[Basis[M_, Ctype[DT_]], np.dtype[DT_]],
        other: Array[Basis[M_, Ctype[DT_]], np.dtype[DT_]],
    ) -> Array[Basis[M_, Ctype[DT_]], np.dtype[DT_]]:
        array = cast("Array[Any, np.dtype[DT_]]", super()).__add__(other)
        if isinstance(other, OperatorList):
            return cast(
                "Any", OperatorList.build(array.basis, array.raw_data).assert_ok()
            )
        return array

    @overload
    def __sub__[M1: OperatorListMetadata, DT_: np.number](
        self: OperatorList[Basis[M1, Ctype[DT_]], np.dtype[DT_]],
        other: OperatorList[Basis[M1, Ctype[DT_]], np.dtype[DT_]],
    ) -> OperatorList[Basis[M1, Ctype[DT_]], np.dtype[DT_]]: ...
    @overload
    def __sub__[M_: BasisMetadata, DT_: np.number](
        self: Array[Basis[M_, Ctype[DT_]], np.dtype[DT_]],
        other: Array[Basis[M_, Ctype[DT_]], np.dtype[DT_]],
    ) -> Array[Basis[M_, Ctype[DT_]], np.dtype[DT_]]: ...
    @override
    def __sub__[M_: BasisMetadata, DT_: np.number](
        self: Array[Basis[M_, Ctype[DT_]], np.dtype[DT_]],
        other: Array[Basis[M_, Ctype[DT_]], np.dtype[DT_]],
    ) -> Array[Basis[M_, Ctype[DT_]], np.dtype[DT_]]:
        array = cast("Array[Any, np.dtype[DT_]]", super()).__sub__(other)
        if isinstance(other, OperatorList):
            return cast(
                "Any", OperatorList.build(array.basis, array.raw_data).assert_ok()
            )

        return array

    @override
    def __mul__[M_: OperatorListMetadata, DT_: np.number](
        self: OperatorList[Basis[M_, Ctype[DT_]], np.dtype[DT_]],
        other: float,
    ) -> OperatorList[Basis[M_, Ctype[DT_]], np.dtype[DT_]]:
        # TODO: always support complex numbers  # noqa: FIX002
        out = cast("Array[Any, np.dtype[DT_]]", super()).__mul__(other)
        return OperatorList.build(out.basis, out.raw_data).assert_ok()


type SuperOperatorMetadata[M: BasisMetadata = BasisMetadata] = OperatorMetadata[
    OperatorMetadata[M],
]
type SuperOperatorBasis[
    M: BasisMetadata = BasisMetadata,
    CT: Ctype[Never] = Ctype[Never],
] = Basis[SuperOperatorMetadata[M], CT]

type SuperOperator[B: SuperOperatorBasis, DT: np.dtype[np.generic]] = Operator[B, DT]
