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
from slate_core import (
    basis as _basis,
)
from slate_core.basis import (
    AsUpcast,
    Basis,
    TupleBasisLike,
    TupleBasisLike2D,
    are_dual_shapes,
)
from slate_core.linalg import into_diagonal
from slate_core.metadata import BasisMetadata, NestedLength

from slate_quantum.state._state import State, StateList, StateWithMetadata

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from slate_core.array import ArrayWithMetadata


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


def operator_basis[B: Basis](
    basis: B,
) -> TupleBasis[tuple[B, B], None]:
    return TupleBasis((basis, basis.dual_basis()))


class Operator[B: OperatorBasis, DT: np.dtype[np.generic]](Array[B, DT]):
    """Represents an operator in a quantum system."""

    def __init__(self, basis: B, data: np.ndarray[Any, DT]) -> None:
        _assert_operator_basis(basis)
        super().__init__(basis, data)

    @overload
    def with_basis[B1_: OperatorBasis](self, basis: B1_) -> Operator[B1_, DT]: ...
    @overload
    def with_basis[B1_: Basis](self, basis: B1_) -> Array[B1_, DT]: ...
    @override
    def with_basis[B1_: Basis](self, basis: B1_) -> Array[B1_, DT]:
        basis.ctype.assert_supports_dtype(self.dtype)
        assert basis.metadata() == self.basis.metadata()
        new_data = self.basis.__convert_vector_into__(self.raw_data, basis).ok()  # type: ignore[return-value]
        return Operator(basis, new_data)  # type: ignore[return-value]

    @overload
    def __add__[M1: OperatorMetadata, DT_: np.number](
        self: Operator[Basis[M1], np.dtype[DT_]],
        other: Operator[Basis[M1], np.dtype[DT_]],
    ) -> Operator[Basis[M1, Ctype[DT_]], np.dtype[DT_]]: ...
    @overload
    def __add__[M_: BasisMetadata, T: np.number](
        self: ArrayWithMetadata[M_, np.dtype[T]],
        other: ArrayWithMetadata[M_, np.dtype[T]],
    ) -> Array[Basis[M_, Ctype[T]], np.dtype[T]]: ...
    @overload
    def __add__[M1: OperatorMetadata, DT_: np.number](
        self: Operator[Basis[M1], np.dtype[DT_]], other: complex
    ) -> Operator[Basis[M1, Ctype[DT_]], np.dtype[np.number]]: ...
    @overload
    def __add__[M_: BasisMetadata, T: np.number](
        self: ArrayWithMetadata[M_, np.dtype[T]],
        other: complex,
    ) -> Array[Basis[M_, Ctype[T]], np.dtype[np.number]]: ...
    @override
    def __add__[M_: BasisMetadata, T: np.number](  # type: ignore bad overload
        self: ArrayWithMetadata[M_, np.dtype[T]],
        other: ArrayWithMetadata[M_, np.dtype[T]] | complex,
    ) -> Any:
        array = cast("Array[Any, np.dtype[T]]", super()).__add__(other)
        if isinstance(other, (Operator, complex, int, float)):
            return cast("Any", Operator(array.basis, array.raw_data))
        return array

    @overload
    def __sub__[M1: OperatorMetadata, DT_: np.number](
        self: Operator[Basis[M1], np.dtype[DT_]],
        other: Operator[Basis[M1], np.dtype[DT_]],
    ) -> Operator[Basis[M1, Ctype[DT_]], np.dtype[DT_]]: ...
    @overload
    def __sub__[M1: BasisMetadata, DT_: np.number](
        self: Array[Basis[M1], np.dtype[DT_]],
        other: Array[Basis[M1], np.dtype[DT_]],
    ) -> Array[Basis[M1, Ctype[DT_]], np.dtype[DT_]]: ...
    @overload
    def __sub__[M1: OperatorMetadata, DT_: np.number](
        self: Operator[Basis[M1], np.dtype[DT_]], other: complex
    ) -> Operator[Basis[M1, Ctype[DT_]], np.dtype[np.number]]: ...
    @overload
    def __sub__[M1: BasisMetadata, DT_: np.number](
        self: Array[Basis[M1], np.dtype[DT_]], other: complex
    ) -> Array[Basis[M1, Ctype[DT_]], np.dtype[np.number]]: ...
    @override
    def __sub__[M_: BasisMetadata, T: np.number](  # type: ignore bad overload
        self: ArrayWithMetadata[M_, np.dtype[T]],
        other: ArrayWithMetadata[M_, np.dtype[T]] | complex,
    ) -> Any:
        array = cast("Array[Any, np.dtype[T]]", super()).__sub__(other)
        if isinstance(other, (Operator, complex, int, float)):
            return cast("Any", Operator(array.basis, array.raw_data))

        return array

    @override
    def __mul__[M_: OperatorMetadata, DT_: np.number](
        self: Operator[Basis[M_], np.dtype[DT_]],
        other: complex,
    ) -> Operator[Basis[M_], np.dtype[np.number]]:
        out = cast("Array[Any, np.dtype[DT_]]", super()).__mul__(other)
        return Operator(out.basis, out.raw_data)

    def as_diagonal[M_: BasisMetadata](
        self: Operator[OperatorBasis[M_], np.dtype[np.complexfloating]],
    ) -> Array[Basis[M_, Ctype[np.complexfloating]], np.dtype[np.complexfloating]]:
        diagonal = into_diagonal(self)
        inner = AsUpcast(
            diagonal.basis.inner.children[1], self.basis.metadata().children[1]
        ).resolve_ctype()
        return Array(inner, diagonal.raw_data)

    @override
    def as_type[
        M_: BasisMetadata,
        DT_: np.number,
    ](
        self: Operator[OperatorBasis[M_], np.dtype[np.generic]],
        ty: type[DT_],
    ) -> Operator[OperatorBasis[M_, Ctype[DT_]], np.dtype[DT_]]:
        return cast(
            "Operator[OperatorBasis[M_, Ctype[DT_]], np.dtype[DT_]]",
            super().as_type(ty),
        )


type OperatorWithMetadata[M: BasisMetadata, DT: np.dtype[np.generic]] = Operator[
    OperatorBasis[M], DT
]


def _assert_operator_list_basis(basis: Basis) -> None:
    is_dual = basis.is_dual
    if isinstance(is_dual, bool):
        msg = "Basis is not 2d"
        raise TypeError(msg)

    _assert_operator_basis(_basis.as_tuple(basis).children[1])


def expectation[M: BasisMetadata](
    operator: Operator[OperatorBasis[M], np.dtype[np.complexfloating]],
    state: StateWithMetadata[M],
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
    basis = _basis.as_tuple(states.basis).children[1]
    return linalg.einsum(
        "(a i'),(i j'),(a j) -> a",
        states.with_state_basis(basis.dual_basis()),
        operator,
        states,
    )


def apply[M: BasisMetadata](
    operator: Operator[Basis[OperatorMetadata[M]], np.dtype[np.complexfloating]],
    state: StateWithMetadata[M],
) -> StateWithMetadata[M]:
    """Apply an operator to a state."""
    out = linalg.einsum("(i j'),j -> i", operator, state)
    return State(out.basis, out.raw_data)


def apply_to_each[M0: BasisMetadata, M: BasisMetadata](
    operator: Operator[Basis[OperatorMetadata[M]], np.dtype[np.complexfloating]],
    states: StateList[TupleBasisLike[tuple[M0, M], None]],
) -> StateList[TupleBasisLike[tuple[M0, M], None]]:
    """Apply an operator to a state."""
    out = linalg.einsum("(i j'),(k j) -> (k i)", operator, states)
    return StateList(out.basis, out.raw_data)


type OperatorListMetadata[
    M0: BasisMetadata = BasisMetadata,
    M1: OperatorMetadata = OperatorMetadata,
] = TupleMetadata[tuple[M0, M1], None]

type OperatorListBasis[
    M0: BasisMetadata = BasisMetadata,
    M1: OperatorMetadata = OperatorMetadata,
] = Basis[OperatorListMetadata[M0, M1]]


class OperatorList[
    B: OperatorListBasis = OperatorListBasis,
    DT: np.dtype[np.generic] = np.dtype[np.generic],
](Array[B, DT]):
    """Represents an operator in a quantum system."""

    def __init__(self, basis: B, data: np.ndarray[Any, DT]) -> None:
        _assert_operator_list_basis(basis)
        super().__init__(basis, data)

    @property
    @override
    def fundamental_shape(self) -> tuple[NestedLength, NestedLength]:
        return cast("tuple[NestedLength, NestedLength]", super().fundamental_shape)

    @overload
    def with_basis[B1_: OperatorListBasis](
        self, basis: B1_
    ) -> OperatorList[B1_, DT]: ...
    @overload
    def with_basis[B1_: Basis](self, basis: B1_) -> Array[B1_, DT]: ...

    @override
    def with_basis[B1_: Basis](self, basis: B1_) -> Array[B1_, DT]:
        basis.ctype.assert_supports_dtype(self.dtype)
        assert basis.metadata() == self.basis.metadata()
        new_data = self.basis.__convert_vector_into__(self.raw_data, basis).ok()  # type: ignore[return-value]
        return OperatorList(basis, new_data)  # type: ignore[return-value]

    def with_operator_basis[
        M: BasisMetadata,
        M1: OperatorMetadata,
        B1: OperatorBasis,
    ](  # B1: B
        self: OperatorList[OperatorListBasis[M, M1], DT], basis: B1
    ) -> OperatorList[
        AsUpcast[
            TupleBasis[tuple[Basis[M], B1], None],
            TupleMetadata[tuple[M, M1], None],
        ],
        DT,
    ]:
        """Get the Operator with the operator basis set to basis."""
        lhs_basis = _basis.as_tuple(self.basis).children[0]
        out_basis = AsUpcast(TupleBasis((lhs_basis, basis)), self.basis.metadata())
        return self.with_basis(out_basis)

    @overload
    def with_list_basis[M: OperatorMetadata, M1: BasisMetadata, B1: Basis](
        self: OperatorList[
            AsUpcast[
                TupleBasis[tuple[Basis[M1], B], None],
                TupleMetadata[tuple[M1, M], None],
            ],
            DT,
        ],
        basis: B1,
    ) -> OperatorList[
        AsUpcast[
            TupleBasis[tuple[B1, B], None],
            TupleMetadata[tuple[M1, M], None],
        ],
        DT,
    ]: ...

    @overload
    def with_list_basis[M: OperatorMetadata, M1: BasisMetadata, B1: Basis](
        self: OperatorList[OperatorListBasis[M1, M], DT], basis: B1
    ) -> OperatorList[
        AsUpcast[
            TupleBasis[tuple[B1, Basis[M]], None],
            TupleMetadata[tuple[M1, M], None],
        ],
        DT,
    ]: ...

    def with_list_basis[M: OperatorMetadata, M1: BasisMetadata, B1: Basis](
        self: OperatorList[OperatorListBasis[M1, M], DT], basis: B1
    ) -> OperatorList[
        AsUpcast[
            TupleBasis[tuple[B1, Any], None],
            TupleMetadata[tuple[M1, M], None],
        ],
        DT,
    ]:
        """Get the Operator with the operator basis set to basis."""
        rhs_basis = _basis.as_tuple(self.basis).children[1]
        out_basis = AsUpcast(TupleBasis((basis, rhs_basis)), self.basis.metadata())
        return self.with_basis(out_basis)

    @override
    def __iter__[M1_: OperatorMetadata](  # type: ignore bad overload
        self: OperatorList[OperatorListBasis[Any, M1_]], /
    ) -> Iterator[Operator[Basis[M1_], DT]]:
        return (Operator(a.basis, a.raw_data) for a in super().__iter__())  # type: ignore cant infer

    @overload
    def __getitem__[M1_: OperatorMetadata](
        self: OperatorList[TupleBasisLike2D[tuple[Any, M1_]]],
        index: tuple[int, slice[None]],
    ) -> Operator[Basis[M1_], DT]: ...

    @overload
    def __getitem__[M1_: OperatorMetadata, I: slice | tuple[array.NestedIndex, ...]](
        self: OperatorList[TupleBasisLike2D[tuple[Any, M1_]], DT],
        index: tuple[I, slice[None]],
    ) -> OperatorList[TupleBasisLike2D[tuple[Any, M1_]], DT]: ...

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
            return Operator(out.basis, out.raw_data)
        if isinstance(index, tuple) and index[1] == slice(None):
            out = cast(
                "Array[Basis[Any, Ctype[np.generic]], np.dtype[np.complexfloating]]",
                out,
            )
            return OperatorList(out.basis, out.raw_data)
        return out

    @staticmethod
    def from_operators[B_: OperatorBasis, DT_: np.dtype[np.generic]](
        iter_: Iterable[Operator[B_, DT_]],
    ) -> OperatorList[
        AsUpcast[
            TupleBasis[tuple[FundamentalBasis[SimpleMetadata], B_], None],
            OperatorListMetadata[SimpleMetadata, OperatorMetadata],
        ],
        DT_,
    ]:
        states = list(iter_)
        assert all(x.basis == states[0].basis for x in states)

        list_basis = FundamentalBasis.from_size(len(states))
        state_basis = TupleBasis((list_basis, states[0].basis)).upcast()
        return OperatorList(
            state_basis,
            cast("np.ndarray[Any, DT_]", np.array([x.raw_data for x in states])),
        )

    @overload
    def __add__[M1: OperatorListMetadata, DT_: np.number](
        self: OperatorList[Basis[M1], np.dtype[DT_]],
        other: OperatorList[Basis[M1], np.dtype[DT_]],
    ) -> OperatorList[Basis[M1, Ctype[DT_]], np.dtype[DT_]]: ...
    @overload
    def __add__[M_: BasisMetadata, DT_: np.number](
        self: Array[Basis[M_], np.dtype[DT_]],
        other: Array[Basis[M_], np.dtype[DT_]],
    ) -> Array[Basis[M_, Ctype[DT_]], np.dtype[DT_]]: ...
    @overload
    def __add__[M_: OperatorListMetadata, T: np.number](
        self: OperatorList[Basis[M_], np.dtype[T]],
        other: complex,
    ) -> Array[Basis[M_, Ctype[T]], np.dtype[np.number]]: ...
    @overload
    def __add__[M_: BasisMetadata, T: np.number](
        self: ArrayWithMetadata[M_, np.dtype[T]],
        other: complex,
    ) -> Array[Basis[M_, Ctype[T]], np.dtype[np.number]]: ...

    @override
    def __add__[M_: BasisMetadata, DT_: np.number](  # type: ignore bad overload
        self: Array[Basis[M_], np.dtype[DT_]],
        other: Array[Basis[M_], np.dtype[DT_]],
    ) -> Array[Basis[M_, Ctype[DT_]], np.dtype[DT_]]:
        array = cast("Array[Any, np.dtype[DT_]]", super()).__add__(other)
        if isinstance(other, (OperatorList, complex, int, float)):
            return cast("Any", OperatorList(array.basis, array.raw_data))
        return array

    @overload
    def __sub__[M1: OperatorListMetadata, DT_: np.number](
        self: OperatorList[Basis[M1], np.dtype[DT_]],
        other: OperatorList[Basis[M1], np.dtype[DT_]],
    ) -> OperatorList[Basis[M1, Ctype[DT_]], np.dtype[DT_]]: ...
    @overload
    def __sub__[M_: BasisMetadata, DT_: np.number](
        self: Array[Basis[M_], np.dtype[DT_]],
        other: Array[Basis[M_], np.dtype[DT_]],
    ) -> Array[Basis[M_, Ctype[DT_]], np.dtype[DT_]]: ...
    @overload
    def __sub__[M_: OperatorListMetadata, T: np.number](
        self: OperatorList[Basis[M_], np.dtype[T]],
        other: complex,
    ) -> Array[Basis[M_, Ctype[T]], np.dtype[np.number]]: ...
    @overload
    def __sub__[M_: BasisMetadata, T: np.number](
        self: ArrayWithMetadata[M_, np.dtype[T]],
        other: complex,
    ) -> Array[Basis[M_, Ctype[T]], np.dtype[np.number]]: ...
    @override
    def __sub__[M_: BasisMetadata, DT_: np.number](  # type: ignore bad overload
        self: Array[Basis[M_], np.dtype[DT_]],
        other: Array[Basis[M_], np.dtype[DT_]],
    ) -> Array[Basis[M_, Ctype[DT_]], np.dtype[DT_]]:
        array = cast("Array[Any, np.dtype[DT_]]", super()).__sub__(other)
        if isinstance(other, (OperatorList, complex, int, float)):
            return OperatorList(array.basis, array.raw_data)

        return array

    @override
    def __mul__[M_: OperatorListMetadata, DT_: np.number](
        self: OperatorList[Basis[M_], np.dtype[DT_]],
        other: complex,
    ) -> OperatorList[Basis[M_], np.dtype[np.number]]:
        out = cast("Array[Any, np.dtype[DT_]]", super()).__mul__(other)
        return OperatorList(out.basis, out.raw_data)

    @override
    def as_type[
        M_: OperatorListMetadata,
        DT_: np.number,
    ](
        self: OperatorList[Basis[M_], np.dtype[np.generic]],
        ty: type[DT_],
    ) -> OperatorList[Basis[M_, Ctype[DT_]], np.dtype[DT_]]:
        return cast(
            "OperatorList[Basis[M_, Ctype[DT_]], np.dtype[DT_]]",
            super().as_type(ty),
        )


type OperatorListWithMetadata[
    M0: BasisMetadata,
    M1: BasisMetadata,
    DT: np.dtype[np.generic],
] = OperatorList[Basis[OperatorListMetadata[M0, OperatorMetadata[M1]]], DT]


type SuperOperatorMetadata[M: BasisMetadata = BasisMetadata] = OperatorMetadata[
    OperatorMetadata[M],
]
type SuperOperatorBasis[
    M: BasisMetadata = BasisMetadata,
    CT: Ctype[Never] = Ctype[Never],
] = Basis[SuperOperatorMetadata[M], CT]

type SuperOperator[B: SuperOperatorBasis, DT: np.dtype[np.generic]] = Operator[B, DT]
