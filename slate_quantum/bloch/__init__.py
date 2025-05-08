"""Makes use of Bloch's theorem to efficiently calculate eigenstates."""

from __future__ import annotations

from slate_quantum.bloch import build
from slate_quantum.bloch._linalg import (
    BlochEigenstateBasis,
    DiagonalBlochBasis,
    into_diagonal,
)
from slate_quantum.bloch._shifted_basis import BlochShiftedBasis
from slate_quantum.bloch._transposed_basis import (
    BlochStateBasis,
    BlochStateMetadata,
    BlochTransposedBasis,
    BlockDiagonalBasis,
)

__all__ = [
    "BlochEigenstateBasis",
    "BlochShiftedBasis",
    "BlochStateBasis",
    "BlochStateMetadata",
    "BlochTransposedBasis",
    "BlockDiagonalBasis",
    "DiagonalBlochBasis",
    "build",
    "into_diagonal",
]
