"""Makes use of Bloch's theorem to efficiently calculate eigenstates."""

from __future__ import annotations

from slate_quantum.bloch import build
from slate_quantum.bloch._linalg import (
    BlochEigenstateBasis,
    DiagonalBlochBasis,
    into_diagonal,
)
from slate_quantum.bloch._shifted_basis import LegacyBlochShiftedBasis
from slate_quantum.bloch._transposed_basis import LegacyBlochTransposedBasis

__all__ = [
    "BlochEigenstateBasis",
    "DiagonalBlochBasis",
    "LegacyBlochShiftedBasis",
    "LegacyBlochTransposedBasis",
    "build",
    "into_diagonal",
]
