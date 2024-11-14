from __future__ import annotations  # noqa: A005

from typing import Any, Iterator, cast, override

import numpy as np
from slate.array import SlateArray
from slate.basis import Basis
from slate.basis.stacked import as_tuple_basis
from slate.metadata import BasisMetadata, StackedMetadata


class Operator[DT: np.generic, B: Basis[StackedMetadata[BasisMetadata, Any], Any]](
    SlateArray[DT, B]
):
    """Represents an operator in a quantum system."""

    @override
    def with_basis[B1: Basis[Any, Any]](  # B1: B
        self, basis: B1
    ) -> Operator[DT, B1]:
        """Get the Operator with the basis set to basis."""
        return Operator(basis, self.basis.__convert_vector_into__(self.raw_data, basis))

    def __add__[_DT: np.number[Any], M: StackedMetadata[BasisMetadata, Any]](
        self: Operator[_DT, Basis[M, Any]], other: Operator[_DT, Basis[M, Any]]
    ) -> Operator[_DT, Basis[M, Any]]:
        res = self.raw_data + other.with_basis(self.basis).raw_data
        data = cast(np.ndarray[Any, np.dtype[_DT]], res)
        return Operator[_DT, Basis[M, Any]](self.basis, data)

    def __sub__[_DT: np.number[Any], M: StackedMetadata[BasisMetadata, Any]](
        self: Operator[_DT, Basis[M, Any]], other: Operator[_DT, Basis[M, Any]]
    ) -> Operator[_DT, Basis[M, Any]]:
        res = self.raw_data - other.with_basis(self.basis).raw_data
        data = cast(np.ndarray[Any, np.dtype[_DT]], res)
        return Operator[_DT, Basis[M, Any]](self.basis, data)


class OperatorList[DT: np.generic, B: Basis[StackedMetadata[BasisMetadata, Any], Any]](
    SlateArray[DT, B]
):
    """Represents an operator in a quantum system."""

    @override
    def with_basis[B1: Basis[Any, Any]](  # B1: B
        self, basis: B1
    ) -> OperatorList[DT, B1]:
        """Get the Operator with the basis set to basis."""
        return OperatorList(
            basis, self.basis.__convert_vector_into__(self.raw_data, basis)
        )

    def __iter__(self, /) -> Iterator[Operator[DT, Basis[Any, np.complex128]]]:
        as_tuple = self.with_basis(as_tuple_basis(self.basis))
        return (
            Operator[DT, Basis[Any, np.complex128]](as_tuple.basis[1], row)
            for row in as_tuple.raw_data.reshape(as_tuple.basis.shape)
        )

    def __getitem__(self, /, index: int) -> Operator[DT, Basis[Any, np.complex128]]:
        as_tuple = self.with_basis(as_tuple_basis(self.basis))
        return Operator[DT, Basis[Any, np.complex128]](
            as_tuple.basis[1], as_tuple.raw_data.reshape(as_tuple.basis.shape)[index]
        )
