from __future__ import annotations

from typing import TYPE_CHECKING, Any

from slate.metadata import BasisMetadata

if TYPE_CHECKING:
    import numpy as np
    from slate.basis import Basis, DiagonalBasis

    from slate_quantum.bloch.basis import BlochDiagonalBasis, BlochEigenstateBasis
    from slate_quantum.model.operator._operator import Operator


def build_bloch_operator[M: BasisMetadata]() -> (
    Operator[
        np.complex128,
        BlochDiagonalBasis[np.complex128, Basis[Any]],
    ]
):
    """Build the diagonalized Bloch Hamiltonian."""
    msg = "Not yet implemented"
    raise NotImplementedError(msg)


def eigh_bloch_operator[M: BasisMetadata]() -> (
    Operator[
        np.complex128,
        DiagonalBasis[
            np.complex128, BlochEigenstateBasis[M], BlochEigenstateBasis[M], None
        ],
    ]
):
    """Build the diagonalized Bloch Hamiltonian."""
    msg = "Not yet implemented"
    raise NotImplementedError(msg)


def build_diagonal_bloch_operator[M: BasisMetadata]() -> (
    Operator[
        np.complex128,
        DiagonalBasis[
            np.complex128, BlochEigenstateBasis[M], BlochEigenstateBasis[M], None
        ],
    ]
):
    """Build the diagonalized Bloch Hamiltonian."""
    msg = "Not yet implemented"
    raise NotImplementedError(msg)
