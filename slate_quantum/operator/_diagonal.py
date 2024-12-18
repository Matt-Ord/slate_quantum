from __future__ import annotations

from typing import Any, cast

import numpy as np
from slate import Basis, BasisMetadata, StackedMetadata, TupleBasis
from slate import basis as _basis
from slate.basis import (
    DiagonalBasis,
    RecastBasis,
    diagonal_basis,
    fundamental_transformed_tuple_basis_from_metadata,
)
from slate.metadata import AxisDirections, Metadata2D

from slate_quantum.operator._operator import Operator

type RecastDiagonalOperatorBasis[
    M: BasisMetadata,
    DT: np.generic,
    BInner: Basis[BasisMetadata, Any] = Basis[M, DT],  # noqa: E251
    BOuter: Basis[BasisMetadata, Any] = Basis[M, DT],  # noqa: E251
] = RecastBasis[
    Metadata2D[M, M, None], M, DT, DiagonalBasis[Any, BInner, BInner, None], BOuter
]


def recast_diagonal_basis[
    M: BasisMetadata,
    DT: np.generic,
    BInner: Basis[BasisMetadata, Any] = Basis[M, DT],
    BOuter: Basis[BasisMetadata, Any] = Basis[M, DT],
](
    inner_basis: BInner, outer_basis: BOuter
) -> RecastDiagonalOperatorBasis[Any, Any, BInner, BOuter]:
    return cast(
        "RecastDiagonalOperatorBasis[M, DT, BInner, BOuter]",
        RecastBasis(
            diagonal_basis((inner_basis, inner_basis.dual_basis())),
            inner_basis.dual_basis(),
            outer_basis.dual_basis(),
        ),
    )


class DiagonalOperator[
    M: BasisMetadata,
    DT: np.generic,
    BInner: Basis[BasisMetadata, Any] = Basis[M, DT],
    BOuter: Basis[BasisMetadata, Any] = Basis[M, DT],
](Operator[M, DT, RecastDiagonalOperatorBasis[M, DT, BInner, BOuter]]):
    def __init__[
        _M: BasisMetadata,
        _DT: np.generic,
        _BInner: Basis[BasisMetadata, Any] = Basis[_M, _DT],
        _BOuter: Basis[BasisMetadata, Any] = Basis[_M, _DT],
    ](
        self: DiagonalOperator[Any, Any, _BInner, _BOuter],
        inner_basis: _BInner,
        outer_basis: _BOuter,
        raw_data: np.ndarray[Any, np.dtype[DT]],
    ) -> None:
        basis = cast(
            "RecastDiagonalOperatorBasis[M, DT, BInner, BOuter]",
            recast_diagonal_basis(inner_basis, outer_basis),
        )
        super().__init__(basis, raw_data)

    @property
    def inner_recast_basis(self) -> BInner:
        return cast("BInner", self.basis.inner_recast)

    @property
    def outer_recast_basis(self) -> BOuter:
        return self.basis.outer_recast

    def with_outer_basis[
        _M: BasisMetadata,
        _DT: np.generic,
        _BInner: Basis[BasisMetadata, Any] = Basis[_M, Any],
        _BOuter: Basis[BasisMetadata, Any] = Basis[_M, Any],
    ](
        self: DiagonalOperator[_M, _DT, _BInner, Basis[_M, Any]], basis: _BOuter
    ) -> DiagonalOperator[_M, _DT, _BInner, _BOuter]:
        """Get the Potential with the outer recast basis set to basis."""
        return DiagonalOperator(
            self.inner_recast_basis,
            basis,
            self.outer_recast_basis.__convert_vector_into__(self.raw_data, basis),
        )


# TODO: DiagonalOperatorList # noqa: FIX002


class PositionOperator[M: BasisMetadata, E: AxisDirections, DT: np.generic](
    DiagonalOperator[
        StackedMetadata[M, E],
        DT,
        TupleBasis[M, E, DT, StackedMetadata[M, E]],
        Basis[StackedMetadata[M, E], Any],
    ]
):
    def __init__(
        self,
        basis: Basis[StackedMetadata[M, E], Any],
        raw_data: np.ndarray[Any, np.dtype[DT]],
    ) -> None:
        super().__init__(
            _basis.from_metadata(basis.metadata()),
            basis,
            raw_data,
        )


type PositionOperatorBasis[M: BasisMetadata, E, DT: np.generic] = (
    RecastDiagonalOperatorBasis[
        StackedMetadata[M, E],
        DT,
        TupleBasis[M, E, DT, StackedMetadata[M, E]],
        Basis[StackedMetadata[M, E], Any],
    ]
)


def position_operator_basis[M: BasisMetadata, E, DT: np.generic](
    basis: Basis[StackedMetadata[M, E], DT],
) -> PositionOperatorBasis[M, E, DT]:
    return recast_diagonal_basis(
        _basis.from_metadata(basis.metadata(), is_dual=basis.is_dual), basis
    )


class Potential[M: BasisMetadata, E: AxisDirections, DT: np.generic](
    PositionOperator[M, E, DT]
): ...


class MomentumOperator[M: BasisMetadata, E: AxisDirections](
    DiagonalOperator[
        StackedMetadata[M, E],
        np.complexfloating,
        TupleBasis[M, E, np.complexfloating, StackedMetadata[M, E]],
        Basis[StackedMetadata[M, E], np.complexfloating],
    ]
):
    def __init__(
        self,
        basis: Basis[StackedMetadata[M, E], Any],
        raw_data: np.ndarray[Any, np.dtype[np.complexfloating]],
    ) -> None:
        super().__init__(
            fundamental_transformed_tuple_basis_from_metadata(
                basis.metadata(), is_dual=basis.is_dual
            ),
            basis,
            raw_data,
        )


type MomentumOperatorBasis[M: BasisMetadata, E: AxisDirections] = (
    RecastDiagonalOperatorBasis[
        StackedMetadata[M, E],
        np.complexfloating,
        TupleBasis[M, E, np.complexfloating, StackedMetadata[M, E]],
        Basis[StackedMetadata[M, E], np.complexfloating],
    ]
)


def momentum_operator_basis[M: BasisMetadata, E: AxisDirections](
    basis: Basis[StackedMetadata[M, E], np.complexfloating],
) -> MomentumOperatorBasis[M, E]:
    return recast_diagonal_basis(
        fundamental_transformed_tuple_basis_from_metadata(
            basis.metadata(), is_dual=basis.is_dual
        ),
        basis,
    )
