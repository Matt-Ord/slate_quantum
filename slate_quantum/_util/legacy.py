from __future__ import annotations

from typing import Any, overload

import numpy as np
from slate_core import Array, Basis, BasisMetadata, Ctype, TupleBasis, TupleMetadata
from slate_core.basis import (
    BlockDiagonalBasis,
    DiagonalBasis,
    IsotropicBasis,
    RecastBasis,
    TupleBasis2D,
)

type LegacyBasis[M: BasisMetadata, DT: np.generic] = Basis[M, Any]


type LegacyArray[M: BasisMetadata, DT: np.generic, B: Basis = Basis] = Array[
    B, np.dtype[DT]
]


type Metadata2D[M0: BasisMetadata, M1: BasisMetadata, E] = TupleMetadata[
    tuple[M0, M1], E
]

type LegacyTupleBasis[
    M: BasisMetadata,
    E,
    DT: np.generic,
    M1: BasisMetadata = Any,
] = TupleBasis[tuple[Basis[M], ...], E, Ctype[DT]]

type LegacyTupleBasis2D[
    DT: np.generic,
    B0: Basis[Any, Any],
    B1: Basis[Any, Any],
    E,
] = TupleBasis2D[tuple[B0, B1], E, Ctype[DT]]


type StackedMetadata[M: BasisMetadata, E] = TupleMetadata[tuple[M, ...], E]


type LegacyRecastBasis[
    M0: BasisMetadata,
    M1: BasisMetadata,
    DT: np.generic,
    B0: Basis[Any, Any],
    B1: Basis[Any, Any],
] = RecastBasis[Any, Any, Any, Any]
type LegacyDiagonalBasis[
    DT: np.generic,
    B0: Basis[Any, Any],
    B1: Basis[Any, Any],
    E: Any,
] = DiagonalBasis[Any, Any]
type LegacyBlockDiagonalBasis[
    DT: np.generic,
    M: BasisMetadata,
    E: Any,
    B1: Basis[Any, Any],
] = BlockDiagonalBasis[Any, Any]
type LegacyIsotropicBasis[
    M: BasisMetadata,
    B0: Basis[Any, Any],
    B1: Basis[Any, Any],
    E,
] = IsotropicBasis[Any, Any]


@overload
def tuple_basis[C: tuple[Basis, ...], E](
    children: C, extra_metadata: E
) -> TupleBasis[C, E, Any]: ...


@overload
def tuple_basis[C: tuple[Basis, ...], E](children: C) -> TupleBasis[C, None, Any]: ...


def tuple_basis[C: tuple[Basis, ...], E](
    children: C, extra_metadata: Any = None
) -> TupleBasis[C, Any, Any]:
    """Build a VariadicTupleBasis from a tuple."""
    return TupleBasis(children, extra_metadata)


@overload
def diagonal_basis[B0: Basis[Any, Any], B1: Basis[Any, Any]](
    children: tuple[B0, B1], extra_metadata: None = None
) -> LegacyDiagonalBasis[Any, B0, B1, None]: ...


@overload
def diagonal_basis[B0: Basis[Any, Any], B1: Basis[Any, Any], E](
    children: tuple[B0, B1], extra_metadata: E
) -> LegacyDiagonalBasis[Any, B0, B1, E]: ...


def diagonal_basis[B0: Basis[Any, Any], B1: Basis[Any, Any], E](
    children: tuple[B0, B1], extra_metadata: E | None = None
) -> LegacyDiagonalBasis[Any, B0, B1, E | None]:
    """Build a VariadicTupleBasis from a tuple."""
    return DiagonalBasis(TupleBasis(children, extra_metadata))
