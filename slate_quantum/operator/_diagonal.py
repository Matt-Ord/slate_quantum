from typing import Any, Never

import numpy as np
from slate_core import (
    Basis,
    BasisMetadata,
    Ctype,
    FundamentalBasis,
    SimpleMetadata,
    TupleBasis,
    TupleMetadata,
)
from slate_core import basis as _basis
from slate_core.basis import AsUpcast, DiagonalBasis, RecastBasis, TupleBasisLike
from slate_core.metadata import (
    AxisDirections,
)

from slate_quantum.operator._operator import (
    Operator,
    OperatorList,
    OperatorListMetadata,
    OperatorMetadata,
)

type DiagonalOperatorBasis[
    BInner: Basis = Basis,
    BOuter: Basis = Basis,
    CT: Ctype[Never] = Ctype[Never],
    M: OperatorMetadata = OperatorMetadata,
] = AsUpcast[
    RecastBasis[
        AsUpcast[
            DiagonalBasis[TupleBasis[tuple[BInner, BInner], None, CT]],
            OperatorMetadata,
            CT,
        ],
        BInner,
        BOuter,
        CT,
    ],
    M,
    CT,
]

type DiagonalOperatorBasisWithMetadata[
    M: BasisMetadata,
    CT: Ctype[Never] = Ctype[Never],
] = DiagonalOperatorBasis[Basis[M], Basis[M], CT, OperatorMetadata[M]]

type DiagonalOperatorListBasis[M0: SimpleMetadata, M: BasisMetadata] = AsUpcast[
    TupleBasis[
        tuple[
            FundamentalBasis[M0],
            DiagonalOperatorBasisWithMetadata[M],
        ],
        None,
    ],
    OperatorListMetadata[M0, OperatorMetadata[M]],
]

type DiagonalOperatorList[
    M0: SimpleMetadata,
    M: BasisMetadata,
    DT: np.dtype[np.generic] = np.dtype[np.generic],
] = OperatorList[DiagonalOperatorListBasis[M0, M], DT]


def recast_diagonal_basis[
    BInner: Basis,
    BOuter: Basis,
](inner_recast: BInner, outer_recast: BOuter) -> DiagonalOperatorBasis[BInner, BOuter]:
    inner = DiagonalBasis(
        TupleBasis((inner_recast, inner_recast.dual_basis()))
    ).upcast()
    recast = RecastBasis(inner, inner_recast.dual_basis(), outer_recast.dual_basis())
    return AsUpcast(recast, inner.metadata())


def recast_diagonal_basis_with_metadata[M: BasisMetadata](
    inner_recast: Basis[M], outer_recast: Basis[M]
) -> DiagonalOperatorBasisWithMetadata[M]:
    inner = DiagonalBasis(
        TupleBasis((inner_recast, inner_recast.dual_basis()))
    ).upcast()
    recast = RecastBasis(inner, inner_recast.dual_basis(), outer_recast.dual_basis())
    return AsUpcast(recast, inner.metadata())


type DiagonalOperator[B: DiagonalOperatorBasis, DT: np.dtype[np.generic]] = Operator[
    B, DT
]
type DiagonalOperatorWithMetadata[M: BasisMetadata, DT: np.dtype[np.generic]] = (
    DiagonalOperator[DiagonalOperatorBasisWithMetadata[M], DT]
)


def with_outer_basis[BInner: Basis, B: Basis, DT: np.dtype[np.generic]](
    operator: DiagonalOperator[DiagonalOperatorBasis[BInner, Any], DT], outer_basis: B
) -> Operator[DiagonalOperatorBasis[BInner, B], DT]:
    """Create a new operator with the specified outer basis."""
    basis = recast_diagonal_basis(operator.basis.inner.inner_recast, outer_basis)
    return operator.with_basis(basis)


type PositionOperatorBasis[
    M: BasisMetadata = BasisMetadata,
    E = Any,
    CT: Ctype[Never] = Ctype[Never],
] = DiagonalOperatorBasis[
    AsUpcast[
        TupleBasis[tuple[Basis[M, Ctype[np.generic]], ...], E],
        TupleMetadata[tuple[M, ...], E],
    ],
    Basis[TupleMetadata[tuple[M, ...], E]],
    CT,
    OperatorMetadata[TupleMetadata[tuple[M, ...], E]],
]


def _assert_position_ty[M: BasisMetadata, E](  # type: ignore this is just a type test
    basis: PositionOperatorBasis[M, E],
) -> DiagonalOperatorBasisWithMetadata[TupleMetadata[tuple[M, ...], E]]:
    return basis


type PositionOperatorListBasis[M0: SimpleMetadata, M: BasisMetadata, E = Any] = Basis[
    OperatorListMetadata[M0, OperatorMetadata[TupleMetadata[tuple[M, ...], E]]]
]


def position_list_basis_as_diagonal[M0: SimpleMetadata, M: BasisMetadata, E](
    basis: PositionOperatorListBasis[M0, M, E],
) -> DiagonalOperatorListBasis[M0, TupleMetadata[tuple[M, ...], E]]:
    return basis  # type: ignore[return-value] this should be allowed ...


def position_list_as_diagonal[
    M0: SimpleMetadata,
    M: BasisMetadata,
    E,
    DT: np.dtype[np.generic],
](
    operator: PositionOperatorList[M0, M, E, DT],
) -> DiagonalOperatorList[M0, TupleMetadata[tuple[M, ...], E], DT]:
    return OperatorList(
        position_list_basis_as_diagonal(operator.basis), operator.raw_data
    )


type PositionOperator[B: PositionOperatorBasis, DT: np.dtype[np.generic]] = (
    DiagonalOperator[B, DT]
)
type PositionOperatorList[
    M0: SimpleMetadata,
    M: BasisMetadata,
    E = Any,
    DT: np.dtype[np.generic] = np.dtype[np.generic],
] = OperatorList[PositionOperatorListBasis[M0, M, E], DT]

type Potential[
    M: BasisMetadata,
    E: AxisDirections,
    CT: Ctype[Never] = Ctype[Never],
    DT: np.dtype[np.generic] = np.dtype[np.complexfloating],
] = PositionOperator[PositionOperatorBasis[M, E, CT], DT]


def position_operator_basis[M: BasisMetadata, E, CT: Ctype[Never]](
    basis: TupleBasisLike[tuple[M, ...], E, CT],
) -> PositionOperatorBasis[M, E]:
    inner_recast = _basis.from_metadata(
        basis.metadata(), is_dual=basis.is_dual
    ).upcast()
    recast = recast_diagonal_basis(inner_recast, basis).inner
    return AsUpcast(recast, TupleMetadata((basis.metadata(), basis.metadata())))


type MomentumOperator[
    M: BasisMetadata,
    E: AxisDirections,
    CT: Ctype[Never],
    DT: np.dtype[np.generic],
] = DiagonalOperator[
    MomentumOperatorBasis[M, E, CT],
    DT,
]
type MomentumOperatorBasis[
    M: BasisMetadata,
    E,
    CT: Ctype[Never] = Ctype[Never],
] = DiagonalOperatorBasis[
    TupleBasis[tuple[Basis[M, Ctype[np.complexfloating]], ...], E],
    TupleBasisLike[tuple[M, ...], E],
    CT,
    OperatorMetadata[TupleMetadata[tuple[M, ...], E]],
]


def momentum_operator_basis[M: BasisMetadata, E, CT: Ctype[Never]](
    basis: TupleBasisLike[tuple[M, ...], E, CT],
) -> MomentumOperatorBasis[M, E]:
    inner_recast = _basis.transformed_from_metadata(
        basis.metadata(), is_dual=basis.is_dual
    )
    recast = recast_diagonal_basis(inner_recast, basis).inner
    return AsUpcast(recast, TupleMetadata((basis.metadata(), basis.metadata())))
