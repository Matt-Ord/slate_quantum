from __future__ import annotations

from typing import Any

import numpy as np
from slate_core import Array, Basis, BasisMetadata, Ctype, TupleMetadata
from slate_core.basis import TupleBasis2D

type LegacyBasis[M: BasisMetadata, DT: np.generic] = Basis[M, Ctype[DT]]


class LegacyArray[M: BasisMetadata, DT: np.generic, B: Basis = Basis](
    Array[B, np.dtype[DT]]
):
    pass


type Metadata2D[M0: BasisMetadata, M1: BasisMetadata, E] = TupleMetadata[
    tuple[M0, M1], E
]


type LegacyTupleBasis2D[
    DT: np.generic,
    B0: Basis[Any, Any],
    B1: Basis[Any, Any],
    E,
] = TupleBasis2D[tuple[B0, B1], E, Ctype[DT]]


type StackedMetadata[M: BasisMetadata, E] = TupleMetadata[tuple[M, ...], E]
