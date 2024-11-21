from __future__ import annotations  # noqa: A005

from typing import TYPE_CHECKING, Any, Iterator, cast, overload, override

import numpy as np
from slate.array import SlateArray
from slate.basis import (
    Basis,
    FundamentalBasis,
    SplitBasis,
    split_basis,
)
from slate.basis.stacked import TupleBasis, TupleBasis2D, as_tuple_basis, tuple_basis
from slate.metadata import BasisMetadata, Metadata2D, SimpleMetadata

if TYPE_CHECKING:
    from collections.abc import Iterable

    from slate.basis.transformed import TransformDirection


class Operator[
    M: Metadata2D[BasisMetadata, BasisMetadata, Any],
    DT: np.generic,
    B: Basis[Metadata2D[BasisMetadata, BasisMetadata, Any], Any] = Basis[M, DT],
](SlateArray[M, DT, B]):
    """Represents an operator in a quantum system."""

    def __init__[
        DT1: np.generic,
        B1: Basis[Metadata2D[BasisMetadata, BasisMetadata, Any], Any],
    ](
        self: Operator[Any, DT1, B1],
        basis: B1,
        data: np.ndarray[Any, np.dtype[DT]],
    ) -> None:
        super().__init__(cast(Any, basis), cast(Any, data))

    @override
    def with_basis[B1: Basis[Any, Any]](  # B1: B
        self, basis: B1
    ) -> Operator[M, DT, B1]:
        """Get the Operator with the basis set to basis."""
        array_with_basis = super().with_basis(basis)
        return Operator(basis, array_with_basis.raw_data)

    @overload
    def __add__[M1: Metadata2D[BasisMetadata, BasisMetadata, Any], DT1: np.number[Any]](
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
        array = SlateArray[Any, Any].__add__(self, other)
        if isinstance(other, Operator):
            return Operator(array.basis, array.raw_data)

        return array

    @overload
    def __sub__[M1: Metadata2D[BasisMetadata, BasisMetadata, Any], DT1: np.number[Any]](
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
        array = SlateArray[M1, DT1].__sub__(self, other)
        if isinstance(other, Operator):
            return Operator(cast(Any, array.basis), array.raw_data)

        return array

    @override
    def __mul__[M1: Metadata2D[BasisMetadata, BasisMetadata, Any], DT1: np.number[Any]](
        self: SlateArray[M1, DT1],
        other: float,
    ) -> Operator[M1, DT1]:
        out = SlateArray[Any, Any].__mul__(self, other)
        return Operator[Any, Any](out.basis, out.raw_data)


class SplitOperator[M: BasisMetadata, E](
    Operator[
        Metadata2D[M, M, E],
        np.complex128,
        SplitBasis[Basis[M, np.complex128], Basis[M, np.complex128], E],
    ]
):
    """Represents an operator in a split basis."""

    def __init__(  # noqa: PLR0913
        self,
        inner: TupleBasis[M, E, np.complex128],
        a: np.ndarray[Any, np.dtype[np.complex128]],
        b: np.ndarray[Any, np.dtype[np.complex128]],
        c: np.ndarray[Any, np.dtype[np.complex128]],
        d: np.ndarray[Any, np.dtype[np.complex128]],
        *,
        direction: TransformDirection = "forward",
    ) -> None:
        basis = split_basis(
            (inner.children[0], inner.children[1]),
            inner.metadata().extra,
            direction=direction,
        )
        data = np.ones_like(basis.size, dtype=np.complex128)
        n = a.size
        data[0 * n : 1 * n] = a
        data[1 * n : 2 * n] = b
        data[2 * n : 3 * n] = c
        data[3 * n : 4 * n] = d

        super().__init__(basis, data)


class OperatorList[
    M: Metadata2D[BasisMetadata, BasisMetadata, Any],
    DT: np.generic,
    B: Basis[Metadata2D[BasisMetadata, BasisMetadata, Any], Any] = Basis[M, DT],
](SlateArray[M, DT, B]):
    """Represents an operator in a quantum system."""

    def __init__[
        DT1: np.generic,
        B1: Basis[Metadata2D[BasisMetadata, BasisMetadata, None], Any],
    ](
        self: OperatorList[Any, DT1, B1],
        basis: B1,
        data: np.ndarray[Any, np.dtype[DT]],
    ) -> None:
        super().__init__(cast(Any, basis), cast(Any, data))

    @override
    def with_basis[B1: Basis[Any, Any]](  # B1: B
        self, basis: B1
    ) -> OperatorList[M, DT, B1]:
        """Get the Operator with the basis set to basis."""
        return OperatorList(
            basis, self.basis.__convert_vector_into__(self.raw_data, basis)
        )

    @overload
    def with_operator_basis[B0: Basis[Any, Any], B1: Basis[Any, Any]](  # B1: B
        self: OperatorList[Any, Any, TupleBasis2D[Any, B0, Any, None]], basis: B1
    ) -> OperatorList[M, DT, TupleBasis2D[Any, B0, B1, None]]: ...

    @overload
    def with_operator_basis[B1: Basis[Any, Any]](  # B1: B
        self, basis: B1
    ) -> OperatorList[M, DT, TupleBasis2D[Any, Any, B1, None]]: ...

    def with_operator_basis(  # B1: B
        self, basis: Basis[Any, Any]
    ) -> OperatorList[M, DT, Any]:
        """Get the Operator with the operator basis set to basis."""
        final_basis = tuple_basis(
            (as_tuple_basis(self.basis)[0], basis),
            self.basis.metadata().extra,
        )
        return OperatorList(
            final_basis, self.basis.__convert_vector_into__(self.raw_data, final_basis)
        )

    @overload
    def with_list_basis[B0: Basis[Any, Any], B1: Basis[Any, Any]](  # B1: B
        self: OperatorList[Any, Any, TupleBasis2D[Any, Any, B1, None]], basis: B0
    ) -> OperatorList[M, DT, TupleBasis2D[Any, B0, B1, None]]: ...

    @overload
    def with_list_basis[B0: Basis[Any, Any]](  # B1: B
        self, basis: B0
    ) -> OperatorList[M, DT, TupleBasis2D[Any, B0, Any, None]]: ...

    def with_list_basis(  # B1: B
        self, basis: Basis[Any, Any]
    ) -> OperatorList[M, DT, Any]:
        """Get the Operator with the operator basis set to basis."""
        final_basis = tuple_basis(
            (basis, cast(Basis[M, Any], as_tuple_basis(self.basis)[1])),
            self.basis.metadata().extra,
        )
        return OperatorList(
            final_basis, self.basis.__convert_vector_into__(self.raw_data, final_basis)
        )

    def __iter__(self, /) -> Iterator[Operator[M, DT]]:
        as_tuple = self.with_basis(as_tuple_basis(self.basis))
        return (
            Operator[M, DT](cast(Basis[M, DT], as_tuple.basis[1]), row)
            for row in as_tuple.raw_data.reshape(as_tuple.basis.shape)
        )

    def __getitem__(self, /, index: int) -> Operator[M, DT, Basis[M, DT]]:
        as_tuple = self.with_basis(as_tuple_basis(self.basis))
        return Operator[M, DT, Basis[M, DT]](
            cast(Basis[M, DT], as_tuple.basis[1]),
            as_tuple.raw_data.reshape(as_tuple.basis.shape)[index],
        )

    @staticmethod
    def from_operators[
        DT1: np.generic,
        B1: Basis[Metadata2D[BasisMetadata, BasisMetadata, None], Any],
    ](
        _iter: Iterable[Operator[Any, DT1, B1]],
    ) -> OperatorList[
        Metadata2D[Any, Any, None],
        DT1,
        TupleBasis2D[Any, FundamentalBasis[SimpleMetadata], B1, None],
    ]:
        operators = list(_iter)
        assert all(x.basis == operators[0].basis for x in operators)

        list_basis = FundamentalBasis.from_shape((len(operators),))
        return OperatorList(
            tuple_basis((list_basis, operators[0].basis)),
            np.array([x.raw_data for x in operators]),
        )

    @overload
    def __add__[
        M1: Metadata2D[BasisMetadata, BasisMetadata, None],
        DT1: np.number[Any],
    ](
        self: OperatorList[M1, DT1],
        other: OperatorList[M1, DT1],
    ) -> OperatorList[M1, DT1]: ...
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
        array = SlateArray[Any, Any].__add__(self, other)
        if isinstance(other, OperatorList):
            return OperatorList(array.basis, array.raw_data)

        return array

    @overload
    def __sub__[
        M1: Metadata2D[BasisMetadata, BasisMetadata, None],
        DT1: np.number[Any],
    ](
        self: OperatorList[M1, DT1],
        other: OperatorList[M1, DT1],
    ) -> OperatorList[M1, DT1]: ...

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
        array = SlateArray[M1, DT1].__sub__(self, other)
        if isinstance(other, OperatorList):
            return OperatorList(cast(Any, array.basis), array.raw_data)

        return array

    @override
    def __mul__[
        M1: Metadata2D[BasisMetadata, BasisMetadata, None],
        DT1: np.number[Any],
    ](
        self: SlateArray[M1, DT1],
        other: float,
    ) -> OperatorList[M1, DT1]:
        out = SlateArray[Any, Any].__mul__(self, other)
        return OperatorList[Any, Any](out.basis, out.raw_data)
