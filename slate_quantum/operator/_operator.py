from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, overload, override

import numpy as np
from slate import FundamentalBasis, SimpleMetadata, linalg
from slate.array import Array, NestedIndex
from slate.basis import (
    Basis,
    TupleBasis2D,
    are_dual_shapes,
    as_tuple_basis,
    tuple_basis,
)
from slate.linalg import into_diagonal
from slate.metadata import BasisMetadata, Metadata2D, NestedLength

from slate_quantum.state._state import State, StateList

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


def _assert_operator_basis(basis: Basis[BasisMetadata, Any]) -> None:
    is_dual = basis.is_dual
    if isinstance(is_dual, bool):
        msg = "Basis is not 2d"
        raise TypeError(msg)
    assert are_dual_shapes(is_dual[0], is_dual[1])


type OperatorMetadata[M: BasisMetadata = BasisMetadata] = Metadata2D[M, M, None]


def operator_basis[M: BasisMetadata, DT: np.generic](
    basis: Basis[M, DT],
) -> TupleBasis2D[DT, Basis[M, DT], Basis[M, DT], None]:
    return tuple_basis((basis, basis.dual_basis()))


class Operator[
    M: BasisMetadata,
    DT: np.generic,
    B: Basis[Metadata2D[BasisMetadata, BasisMetadata, Any], Any] = Basis[
        Metadata2D[M, M, None], DT
    ],
](Array[Metadata2D[M, M, None], DT, B]):
    """Represents an operator in a quantum system."""

    def __init__[
        DT1: np.generic,
        B1: Basis[Metadata2D[BasisMetadata, BasisMetadata, Any], Any],
    ](
        self: Operator[Any, DT1, B1],
        basis: B1,
        data: np.ndarray[Any, np.dtype[DT]],
    ) -> None:
        super().__init__(cast("Any", basis), cast("Any", data))
        _assert_operator_basis(self.basis)

    @override
    def with_basis[B1: Basis[Any, Any]](  # B1: B
        self, basis: B1
    ) -> Operator[M, DT, B1]:
        """Get the Operator with the basis set to basis."""
        array_with_basis = super().with_basis(basis)
        return Operator(basis, array_with_basis.raw_data)

    @overload
    def __add__[M1: BasisMetadata, DT1: np.number[Any]](
        self: Operator[M1, DT1],
        other: Operator[M1, DT1],
    ) -> Operator[M1, DT1]: ...
    @overload
    def __add__[M1: BasisMetadata, DT1: np.number[Any]](
        self: Array[M1, DT1],
        other: Array[M1, DT1],
    ) -> Array[M1, DT1]: ...
    @override
    def __add__[M1: BasisMetadata, DT1: np.number[Any]](
        self: Array[M1, DT1],
        other: Array[M1, DT1],
    ) -> Array[M1, DT1]:
        array = cast("Array[Any, DT1]", super()).__add__(other)
        if isinstance(other, Operator):
            return cast("Any", Operator(array.basis, array.raw_data))
        return array

    @overload
    def __sub__[M1: BasisMetadata, DT1: np.number[Any]](
        self: Operator[M1, DT1],
        other: Operator[M1, DT1],
    ) -> Operator[M1, DT1]: ...

    @overload
    def __sub__[M1: BasisMetadata, DT1: np.number[Any]](
        self: Array[M1, DT1],
        other: Array[M1, DT1],
    ) -> Array[M1, DT1]: ...

    @override
    def __sub__[M1: BasisMetadata, DT1: np.number[Any]](
        self: Array[M1, DT1],
        other: Array[M1, DT1],
    ) -> Array[M1, DT1]:
        array = cast("Array[Any, DT1]", super()).__sub__(other)
        if isinstance(other, Operator):
            return cast("Any", Operator(array.basis, array.raw_data))

        return array

    @override
    def __mul__[M1: BasisMetadata, DT1: np.number[Any]](
        self: Operator[M1, DT1],
        other: complex,
    ) -> Operator[M1, DT1]:
        # TODO: always support complex numbers  # noqa: FIX002
        out = cast("Array[Any, DT1]", super()).__mul__(cast("float", other))
        return Operator[Any, Any](out.basis, out.raw_data)

    def as_diagonal(self) -> Array[M, np.complexfloating]:
        diagonal = into_diagonal(
            Operator(self.basis, self.raw_data.astype(np.complex128))
        )
        inner = cast("Basis[M, DT]", diagonal.basis.inner[1])
        return Array(inner, diagonal.raw_data)


def _assert_operator_list_basis(basis: Basis[BasisMetadata, Any]) -> None:
    is_dual = basis.is_dual
    if isinstance(is_dual, bool):
        msg = "Basis is not 2d"
        raise TypeError(msg)
    _assert_operator_basis(as_tuple_basis(basis)[1])


def expectation[M: BasisMetadata](
    operator: Operator[M, np.complexfloating],
    state: State[M],
) -> complex:
    """Calculate the expectation value of an operator."""
    return (
        linalg.einsum(
            "i' ,(i j'),j -> ",
            state.with_basis(state.basis.dual_basis()),
            operator,
            state,
        )
        .as_array()
        .item()
    )


def expectation_of_each[M0: BasisMetadata, M: BasisMetadata](
    operator: Operator[M, np.complexfloating],
    states: StateList[M0, M],
) -> Array[M0, np.complexfloating]:
    """Calculate the expectation value of an operator."""
    basis = as_tuple_basis(states.basis)[1]
    return linalg.einsum(
        "(a i'),(i j'),(a j) -> a",
        states.with_state_basis(basis.dual_basis()),
        operator,
        states,
    )


def apply[M: BasisMetadata](
    operator: Operator[M, np.complexfloating],
    state: State[M],
) -> State[M]:
    """Apply an operator to a state."""
    out = linalg.einsum("(i j'),j -> i", operator, state)
    return State(out.basis, out.raw_data)


def apply_to_each[M0: BasisMetadata, M: BasisMetadata](
    operator: Operator[M, np.complexfloating],
    states: StateList[M0, M],
) -> StateList[M0, M]:
    """Apply an operator to a state."""
    out = linalg.einsum("(i j'),(k j) -> (k i)", operator, states)
    return StateList(out.basis, out.raw_data)


OperatorListMetadata = Metadata2D[BasisMetadata, OperatorMetadata, Any]


class OperatorList[
    M0: BasisMetadata,
    M1: BasisMetadata,
    DT: np.generic,
    B: Basis[OperatorListMetadata, Any] = Basis[
        Metadata2D[M0, Metadata2D[M1, M1, None], None], DT
    ],
](Array[Metadata2D[M0, Metadata2D[M1, M1, None], None], DT, B]):
    """Represents an operator in a quantum system."""

    def __init__[
        DT1: np.generic,
        B1: Basis[OperatorListMetadata, Any],
    ](
        self: OperatorList[Any, Any, DT1, B1],
        basis: B1,
        data: np.ndarray[Any, np.dtype[DT1]],
    ) -> None:
        super().__init__(cast("Any", basis), cast("Any", data))
        _assert_operator_list_basis(self.basis)

    @property
    @override
    def fundamental_shape(self) -> tuple[NestedLength, NestedLength]:
        return cast("tuple[NestedLength, NestedLength]", super().fundamental_shape)

    @override
    def with_basis[B1: Basis[Any, Any]](  # B1: B
        self, basis: B1
    ) -> OperatorList[M0, M1, DT, B1]:
        """Get the Operator with the basis set to basis."""
        return OperatorList(
            basis, self.basis.__convert_vector_into__(self.raw_data, basis)
        )

    @overload
    def with_operator_basis[B0: Basis[Any, Any], B1: Basis[Any, Any]](  # B1: B
        self: OperatorList[Any, Any, Any, TupleBasis2D[Any, B0, Any, None]], basis: B1
    ) -> OperatorList[M0, M1, DT, TupleBasis2D[Any, B0, B1, None]]: ...

    @overload
    def with_operator_basis[B1: Basis[Any, Any]](  # B1: B
        self, basis: B1
    ) -> OperatorList[M0, M1, DT, TupleBasis2D[Any, Any, B1, None]]: ...

    def with_operator_basis(  # B1: B
        self, basis: Basis[OperatorMetadata, Any]
    ) -> OperatorList[M0, M1, DT, Any]:
        """Get the Operator with the operator basis set to basis."""
        final_basis = tuple_basis((as_tuple_basis(self.basis)[0], basis))
        return OperatorList(
            final_basis, self.basis.__convert_vector_into__(self.raw_data, final_basis)
        )

    @overload
    def with_list_basis[B0: Basis[Any, Any], B1: Basis[Any, Any]](  # B1: B
        self: OperatorList[Any, Any, Any, TupleBasis2D[Any, Any, B1, None]], basis: B0
    ) -> OperatorList[M0, M1, DT, TupleBasis2D[Any, B0, B1, None]]: ...

    @overload
    def with_list_basis[B0: Basis[Any, Any]](  # B1: B
        self, basis: B0
    ) -> OperatorList[
        M0, M1, DT, TupleBasis2D[Any, B0, Basis[Metadata2D[M1, M1, None], Any], None]
    ]: ...

    def with_list_basis(  # B1: B
        self, basis: Basis[Any, Any]
    ) -> OperatorList[M0, M1, DT, Any]:
        """Get the Operator with the operator basis set to basis."""
        final_basis = tuple_basis((basis, as_tuple_basis(self.basis)[1]))
        return OperatorList(
            final_basis, self.basis.__convert_vector_into__(self.raw_data, final_basis)
        )

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
    def from_operators[
        DT1: np.generic,
        M1_: BasisMetadata,
        B1: Basis[Metadata2D[BasisMetadata, BasisMetadata, None], Any] = Basis[
            Metadata2D[M1_, M1_, None], DT1
        ],
    ](
        iter_: Iterable[Operator[M1_, DT1, B1]],
    ) -> OperatorList[
        SimpleMetadata,
        M1_,
        DT1,
        TupleBasis2D[Any, FundamentalBasis[SimpleMetadata], B1, None],
    ]:
        operators = list(iter_)
        assert all(x.basis == operators[0].basis for x in operators)

        list_basis = FundamentalBasis.from_size(len(operators))
        return OperatorList(
            tuple_basis((list_basis, operators[0].basis)),
            np.array([x.raw_data for x in operators]),
        )

    @overload
    def __add__[
        M0_: BasisMetadata,
        M1_: BasisMetadata,
        DT1: np.number[Any],
    ](
        self: OperatorList[M0_, M1_, DT1],
        other: OperatorList[M0_, M1_, DT1],
    ) -> OperatorList[M0_, M1_, DT1]: ...
    @overload
    def __add__[M0_: BasisMetadata, DT1: np.number[Any]](
        self: Array[M0_, DT1],
        other: Array[M0_, DT1],
    ) -> Array[M0_, DT1]: ...
    @override
    def __add__[M0_: BasisMetadata, DT1: np.number[Any]](
        self: Array[M0_, DT1],
        other: Array[M0_, DT1],
    ) -> Array[M0_, DT1]:
        array = cast("Array[Any, DT1]", super()).__add__(other)
        if isinstance(other, OperatorList):
            return cast("Any", OperatorList(array.basis, array.raw_data))

        return array

    @overload
    def __sub__[
        M0_: BasisMetadata,
        M1_: BasisMetadata,
        DT1: np.number[Any],
    ](
        self: OperatorList[M0_, M1_, DT1],
        other: OperatorList[M0_, M1_, DT1],
    ) -> OperatorList[M0_, M1_, DT1]: ...

    @overload
    def __sub__[M0_: BasisMetadata, DT1: np.number[Any]](
        self: Array[M0_, DT1],
        other: Array[M0_, DT1],
    ) -> Array[M0_, DT1]: ...

    @override
    def __sub__[M0_: BasisMetadata, DT1: np.number[Any]](
        self: Array[M0_, DT1],
        other: Array[M0_, DT1],
    ) -> Array[M0_, DT1]:
        array = cast("Array[Any, DT1]", super()).__sub__(other)
        if isinstance(other, OperatorList):
            return cast("Any", OperatorList(array.basis, array.raw_data))

        return array

    @override
    def __mul__[
        M0_: BasisMetadata,
        M1_: BasisMetadata,
        DT1: np.number[Any],
    ](
        self: OperatorList[M0_, M1_, DT1],
        other: float,
    ) -> OperatorList[M0_, M1_, DT1]:
        out = cast("Array[Any, DT1]", super()).__mul__(other)
        return OperatorList[Any, Any, Any](out.basis, out.raw_data)
