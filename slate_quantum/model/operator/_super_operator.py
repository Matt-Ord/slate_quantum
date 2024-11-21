from __future__ import annotations

from typing import Any, cast, overload, override

import numpy as np
from slate.array import SlateArray
from slate.basis import (
    Basis,
)
from slate.metadata import BasisMetadata, Metadata2D

type SuperOperatorMetadata[
    B0: BasisMetadata,
    B1: BasisMetadata,
    B2: BasisMetadata,
    B3: BasisMetadata,
] = Metadata2D[
    Metadata2D[B0, B1, Any],
    Metadata2D[B2, B3, Any],
    Any,
]

_SuperOperatorMetadata = SuperOperatorMetadata[
    BasisMetadata, BasisMetadata, BasisMetadata, BasisMetadata
]


class SuperOperator[
    M: _SuperOperatorMetadata,
    DT: np.generic,
    B: Basis[
        _SuperOperatorMetadata,
        Any,
    ] = Basis[M, DT],
](SlateArray[M, DT, B]):
    """Represents an operator in a quantum system."""

    def __init__[
        DT1: np.generic,
        B1: Basis[_SuperOperatorMetadata, Any],
    ](
        self: SuperOperator[Any, DT1, B1],
        basis: B1,
        data: np.ndarray[Any, np.dtype[DT]],
    ) -> None:
        super().__init__(cast(Any, basis), cast(Any, data))

    @override
    def with_basis[B1: Basis[Any, Any]](  # B1: B
        self, basis: B1
    ) -> SuperOperator[M, DT, B1]:
        """Get the Operator with the basis set to basis."""
        array_with_basis = super().with_basis(basis)
        return SuperOperator(basis, array_with_basis.raw_data)

    @overload
    def __add__[M1: _SuperOperatorMetadata, DT1: np.number[Any]](
        self: SuperOperator[M1, DT1],
        other: SuperOperator[M1, DT1],
    ) -> SuperOperator[M1, DT1]: ...
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
        if isinstance(other, SuperOperator):
            return SuperOperator(array.basis, array.raw_data)

        return array

    @overload
    def __sub__[M1: _SuperOperatorMetadata, DT1: np.number[Any]](
        self: SuperOperator[M1, DT1],
        other: SuperOperator[M1, DT1],
    ) -> SuperOperator[M1, DT1]: ...

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
        if isinstance(other, SuperOperator):
            return SuperOperator(cast(Any, array.basis), array.raw_data)

        return array

    @override
    def __mul__[M1: _SuperOperatorMetadata, DT1: np.number[Any]](
        self: SlateArray[M1, DT1],
        other: float,
    ) -> SuperOperator[M1, DT1]:
        out = SlateArray[Any, Any].__mul__(self, other)
        return SuperOperator[Any, Any](out.basis, out.raw_data)
