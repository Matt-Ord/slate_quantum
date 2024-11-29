from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, overload, override

import numpy as np
from slate.array import SlateArray
from slate.basis import (
    Basis,
    FundamentalBasis,
    TupleBasis2D,
    are_dual_shapes,
    as_tuple_basis,
    tuple_basis,
)
from slate.linalg import into_diagonal
from slate.metadata import BasisMetadata, Metadata2D, SimpleMetadata

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


def _assert_operator_basis(basis: Basis[BasisMetadata, Any]) -> None:
    is_dual = basis.is_dual
    if isinstance(is_dual, bool):
        msg = "Basis is not 2d"
        raise TypeError(msg)
    assert are_dual_shapes(is_dual[0], is_dual[1])


type OperatorMetadata[M: BasisMetadata = BasisMetadata] = Metadata2D[M, M, None]  # noqa: E251


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
](SlateArray[Metadata2D[M, M, None], DT, B]):
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
        self: SlateArray[M1, DT1],
        other: SlateArray[M1, DT1],
    ) -> SlateArray[M1, DT1]: ...
    @override
    def __add__[M1: BasisMetadata, DT1: np.number[Any]](
        self: SlateArray[M1, DT1],
        other: SlateArray[M1, DT1],
    ) -> SlateArray[M1, DT1]:
        array = cast("SlateArray[Any, DT1]", super()).__add__(other)
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
        self: SlateArray[M1, DT1],
        other: SlateArray[M1, DT1],
    ) -> SlateArray[M1, DT1]: ...

    @override
    def __sub__[M1: BasisMetadata, DT1: np.number[Any]](
        self: SlateArray[M1, DT1],
        other: SlateArray[M1, DT1],
    ) -> SlateArray[M1, DT1]:
        array = cast("SlateArray[Any, DT1]", super()).__sub__(other)
        if isinstance(other, Operator):
            return cast("Any", Operator(array.basis, array.raw_data))

        return array

    @override
    def __mul__[M1: BasisMetadata, DT1: np.number[Any]](
        self: SlateArray[M1, DT1],
        other: float,
    ) -> Operator[M1, DT1]:
        out = cast("SlateArray[Any, DT1]", super()).__mul__(other)
        return Operator[Any, Any](out.basis, out.raw_data)

    def as_diagonal(self) -> SlateArray[M, np.complex128]:
        diagonal = into_diagonal(
            Operator(self.basis, self.raw_data.astype(np.complex128))
        )
        inner = cast("Basis[M, DT]", diagonal.basis.inner[1])
        return SlateArray(inner, diagonal.raw_data)


def _assert_operator_list_basis(basis: Basis[BasisMetadata, Any]) -> None:
    is_dual = basis.is_dual
    if isinstance(is_dual, bool):
        msg = "Basis is not 2d"
        raise TypeError(msg)
    _assert_operator_basis(as_tuple_basis(basis)[1])


OperatorListMetadata = Metadata2D[BasisMetadata, OperatorMetadata, Any]


class OperatorList[
    M0: BasisMetadata,
    M1: BasisMetadata,
    DT: np.generic,
    B: Basis[OperatorListMetadata, Any] = Basis[
        Metadata2D[M0, Metadata2D[M1, M1, None], None], DT
    ],
](SlateArray[Metadata2D[M0, Metadata2D[M1, M1, None], None], DT, B]):
    """Represents an operator in a quantum system."""

    def __init__[
        DT1: np.generic,
        B1: Basis[OperatorListMetadata, Any],
    ](
        self: OperatorList[Any, Any, DT1, B1],
        basis: B1,
        data: np.ndarray[Any, np.dtype[DT]],
    ) -> None:
        super().__init__(cast("Any", basis), cast("Any", data))
        _assert_operator_list_basis(self.basis)

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
    ) -> OperatorList[M0, M1, DT, TupleBasis2D[Any, B0, Any, None]]: ...

    def with_list_basis(  # B1: B
        self, basis: Basis[Any, Any]
    ) -> OperatorList[M0, M1, DT, Any]:
        """Get the Operator with the operator basis set to basis."""
        final_basis = tuple_basis((basis, as_tuple_basis(self.basis)[1]))
        return OperatorList(
            final_basis, self.basis.__convert_vector_into__(self.raw_data, final_basis)
        )

    def __iter__(self, /) -> Iterator[Operator[M1, DT]]:
        as_tuple = self.with_basis(as_tuple_basis(self.basis))
        return (
            Operator[M1, DT](
                cast("Basis[Metadata2D[M1, M1, None], DT]", as_tuple.basis[1]), row
            )
            for row in as_tuple.raw_data.reshape(as_tuple.basis.shape)
        )

    @overload
    def __getitem__[
        _M1: BasisMetadata,
        _DT1: np.generic,
        _B1: Basis[OperatorMetadata, Any] = Basis[Metadata2D[_M1, _M1, None], _DT1],
    ](
        self: OperatorList[Any, _M1, _DT1, TupleBasis2D[Any, Any, _B1, None]],
        /,
        index: int,
    ) -> Operator[_M1, _DT1, _B1]: ...

    @overload
    def __getitem__(self, /, index: int) -> Operator[M1, DT]: ...

    def __getitem__(self, /, index: int) -> Operator[Any, Any, Any]:
        as_tuple = self.with_basis(as_tuple_basis(self.basis))
        return Operator[M1, DT](
            cast("Basis[Metadata2D[M1, M1, None], DT]", as_tuple.basis[1]),
            as_tuple.raw_data.reshape(as_tuple.basis.shape)[index],
        )

    @staticmethod
    def from_operators[
        DT1: np.generic,
        _M1: BasisMetadata,
        _B1: Basis[Metadata2D[BasisMetadata, BasisMetadata, None], Any] = Basis[
            Metadata2D[_M1, _M1, None], DT1
        ],
    ](
        _iter: Iterable[Operator[_M1, DT1, _B1]],
    ) -> OperatorList[
        SimpleMetadata,
        _M1,
        DT1,
        TupleBasis2D[Any, FundamentalBasis[SimpleMetadata], _B1, None],
    ]:
        operators = list(_iter)
        assert all(x.basis == operators[0].basis for x in operators)

        list_basis = FundamentalBasis.from_size(len(operators))
        return OperatorList(
            tuple_basis((list_basis, operators[0].basis)),
            np.array([x.raw_data for x in operators]),
        )

    @overload
    def __add__[
        _M0: BasisMetadata,
        _M1: BasisMetadata,
        DT1: np.number[Any],
    ](
        self: OperatorList[_M0, _M1, DT1],
        other: OperatorList[_M0, _M1, DT1],
    ) -> OperatorList[_M0, _M1, DT1]: ...
    @overload
    def __add__[_M0: BasisMetadata, DT1: np.number[Any]](
        self: SlateArray[_M0, DT1],
        other: SlateArray[_M0, DT1],
    ) -> SlateArray[_M0, DT1]: ...
    @override
    def __add__[_M0: BasisMetadata, DT1: np.number[Any]](
        self: SlateArray[_M0, DT1],
        other: SlateArray[_M0, DT1],
    ) -> SlateArray[_M0, DT1]:
        array = cast("SlateArray[Any, DT1]", super()).__add__(other)
        if isinstance(other, OperatorList):
            return cast("Any", OperatorList(array.basis, array.raw_data))

        return array

    @overload
    def __sub__[
        _M0: BasisMetadata,
        _M1: BasisMetadata,
        DT1: np.number[Any],
    ](
        self: OperatorList[_M0, _M1, DT1],
        other: OperatorList[_M0, _M1, DT1],
    ) -> OperatorList[_M0, _M1, DT1]: ...

    @overload
    def __sub__[_M0: BasisMetadata, DT1: np.number[Any]](
        self: SlateArray[_M0, DT1],
        other: SlateArray[_M0, DT1],
    ) -> SlateArray[_M0, DT1]: ...

    @override
    def __sub__[_M0: BasisMetadata, DT1: np.number[Any]](
        self: SlateArray[_M0, DT1],
        other: SlateArray[_M0, DT1],
    ) -> SlateArray[_M0, DT1]:
        array = cast("SlateArray[Any, DT1]", super()).__sub__(other)
        if isinstance(other, OperatorList):
            return cast("Any", OperatorList(array.basis, array.raw_data))

        return array

    @override
    def __mul__[
        _M0: BasisMetadata,
        _M1: BasisMetadata,
        DT1: np.number[Any],
    ](
        self: OperatorList[_M0, _M1, DT1],
        other: float,
    ) -> OperatorList[_M0, _M1, DT1]:
        out = cast("SlateArray[Any, DT1]", super()).__mul__(other)
        return OperatorList[Any, Any, Any](out.basis, out.raw_data)
