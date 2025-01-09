from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from slate.basis import BlockDiagonalBasis, TupleBasis2D, WrappedBasis
from slate.metadata import Metadata2D

if TYPE_CHECKING:
    from slate.basis import Basis


class ExplicitBlockDiagonalBasis[
    DT: np.generic,
    B0: Basis[Any, Any],
    B1: Basis[Any, Any],
    E,
](
    WrappedBasis[
        Metadata2D[Any, Any, E],
        DT,
        BlockDiagonalBasis[DT, Any, Any, TupleBasis2D[DT, B0, B1, E]],
    ],
): ...


def into_diagonal() -> None: ...
