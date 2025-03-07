from __future__ import annotations

from typing import Any, Never

import numpy as np
from slate_core import Basis, BasisMetadata, TupleBasis, ctype
from slate_core import basis as _basis
from slate_core.basis import AsUpcast, DiagonalBasis, RecastBasis, TupleBasisLike
from slate_core.metadata import AxisDirections

from slate_quantum.operator._operator import (
    Operator,
    OperatorConversion,
    OperatorMetadata,
)

type DiagonalOperatorBasis[
    BInner: Basis = Basis,
    BOuter: Basis = Basis,
    DT: ctype[Never] = ctype[Never],
] = AsUpcast[
    RecastBasis[
        DiagonalBasis[TupleBasis[tuple[BInner, BInner], None]], BInner, BOuter, DT
    ],
    OperatorMetadata,
    DT,
]


def recast_diagonal_basis[
    BInner: Basis = Basis,
    BOuter: Basis = Basis,
](inner_recast: BInner, outer_recast: BOuter) -> DiagonalOperatorBasis[BInner, BOuter]:
    inner = DiagonalBasis(TupleBasis((inner_recast, inner_recast.dual_basis())))
    recast = RecastBasis(inner, inner_recast.dual_basis(), outer_recast.dual_basis())
    return AsUpcast(recast, inner.metadata())


type DiagonalOperator[B: DiagonalOperatorBasis, DT: np.dtype[np.generic]] = Operator[
    B, DT
]
type DiagonalOperatorLike[M: BasisMetadata, DT: np.dtype[np.generic]] = (
    DiagonalOperator[DiagonalOperatorBasis[Basis[M], Basis[M]], DT]
)


def with_outer_basis[BInner: Basis, B: Basis, DT: np.dtype[np.generic]](
    operator: DiagonalOperator[DiagonalOperatorBasis[BInner, Any], DT], outer_basis: B
) -> OperatorConversion[OperatorMetadata, DiagonalOperatorBasis[BInner, B], DT]:
    """Create a new operator with the specified outer basis."""
    # TODO closer bound check
    basis = recast_diagonal_basis(operator.basis.inner.inner_recast, outer_basis)
    return operator.with_basis(basis)


type PositionOperatorBasis[
    M: BasisMetadata = BasisMetadata,
    E = Any,
    DT: ctype[Never] = ctype[Never],
] = DiagonalOperatorBasis[
    TupleBasis[tuple[Basis[M, ctype[np.generic]], ...], E],
    TupleBasisLike[tuple[M, ...], E],
    DT,
]

type PositionOperator[B: PositionOperatorBasis, DT: np.dtype[np.generic]] = (
    DiagonalOperator[B, DT]
)
type Potential[M: BasisMetadata, E: AxisDirections, DT: np.dtype[np.generic]] = (
    PositionOperator[PositionOperatorBasis[M, E], DT]
)


def position_operator_basis[M: BasisMetadata, E, DT: ctype[Never]](
    basis: TupleBasisLike[tuple[M, ...], E, DT],
) -> PositionOperatorBasis[M, E, DT]:
    inner_recast = _basis.from_metadata(basis.metadata(), is_dual=basis.is_dual)
    return recast_diagonal_basis(inner_recast, basis)


type MomentumOperator[M: BasisMetadata, E: AxisDirections, DT: np.dtype[np.generic]] = (
    DiagonalOperator[
        MomentumOperatorBasis[M, E],
        DT,
    ]
)
type MomentumOperatorBasis[M: BasisMetadata, E, DT: ctype[Never] = ctype[Never]] = (
    DiagonalOperatorBasis[
        TupleBasis[tuple[Basis[M, ctype[np.complexfloating]], ...], E],
        TupleBasisLike[tuple[M, ...], E],
        DT,
    ]
)


def momentum_operator_basis[M: BasisMetadata, E, DT: ctype[Never]](
    basis: TupleBasisLike[tuple[M, ...], E, DT],
) -> PositionOperatorBasis[M, E, DT]:
    inner_recast = _basis.transformed_from_metadata(
        basis.metadata(), is_dual=basis.is_dual
    )
    return recast_diagonal_basis(inner_recast, basis)
