from __future__ import annotations

from typing import Any, cast, overload, override

import numpy as np
from slate.array import SlateArray
from slate.basis import (
    Basis,
)
from slate.metadata import BasisMetadata, Metadata2D

from slate_quantum.model.operator._operator import Operator, OperatorMetadata

type SuperOperatorMetadata[M: BasisMetadata = BasisMetadata] = OperatorMetadata[  # noqa: E251
    OperatorMetadata[M],
]


class SuperOperator[
    M: BasisMetadata,
    DT: np.generic,
    B: Basis[
        SuperOperatorMetadata,
        Any,
    ] = Basis[SuperOperatorMetadata[M], DT],
](Operator[Metadata2D[M, M, None], DT, B]):
    """Represents an operator in a quantum system."""

    def __init__[
        DT1: np.generic,
        B1: Basis[SuperOperatorMetadata, Any],
    ](
        self: SuperOperator[Any, DT1, B1],
        basis: B1,
        data: np.ndarray[Any, np.dtype[DT]],
    ) -> None:
        super().__init__(cast("Any", basis), cast("Any", data))

    @override
    def with_basis[B1: Basis[Any, Any]](  # B1: B
        self, basis: B1
    ) -> SuperOperator[M, DT, B1]:
        """Get the Operator with the basis set to basis."""
        array_with_basis = super().with_basis(basis)
        return SuperOperator(basis, array_with_basis.raw_data)

    @overload
    def __add__[M1: BasisMetadata, DT1: np.number[Any]](
        self: SuperOperator[M1, DT1],
        other: SuperOperator[M1, DT1],
    ) -> SuperOperator[M1, DT1]: ...
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
    def __add__[M1: BasisMetadata, DT1: np.number[Any]](  # type: ignore overload
        self: Any,
        other: Any,
    ) -> Any:
        array = Operator[Any, np.number[Any]].__add__(self, other)
        if isinstance(other, SuperOperator):
            return cast("Any", SuperOperator(array.basis, array.raw_data))

        return array

    @overload
    def __sub__[M1: BasisMetadata, DT1: np.number[Any]](
        self: SuperOperator[M1, DT1],
        other: SuperOperator[M1, DT1],
    ) -> SuperOperator[M1, DT1]: ...
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
    def __sub__[M1: BasisMetadata, DT1: np.number[Any]](  # type: ignore overload
        self: Any,
        other: Any,
    ) -> Any:
        array = Operator[Any, DT1].__sub__(self, other)
        if isinstance(other, SuperOperator):
            return cast("Any", SuperOperator(array.basis, array.raw_data))

        return array

    @override
    def __mul__[M1: BasisMetadata, DT1: np.number[Any]](
        self: SuperOperator[M1, DT1],
        other: float,
    ) -> SuperOperator[M1, DT1]:
        out = SlateArray[Any, Any].__mul__(self, other)
        return SuperOperator[Any, Any](out.basis, out.raw_data)
