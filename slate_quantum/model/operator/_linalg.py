from __future__ import annotations

from typing import TYPE_CHECKING, Any

from slate.linalg import eig, eigh
from slate.metadata import BasisMetadata, StackedMetadata

from slate_quantum.model.operator._operator import Operator

if TYPE_CHECKING:
    import numpy as np
    from slate.basis import Basis, DiagonalBasis
    from slate.explicit_basis import ExplicitBasis, ExplicitUnitaryBasis


def eig_operator[M: BasisMetadata](
    operator: Operator[np.complex128, Basis[StackedMetadata[M, Any], np.complex128]],
) -> Operator[
    np.complex128,
    DiagonalBasis[
        np.complex128,
        ExplicitBasis[M, Any],
        ExplicitBasis[M, Any],
        Any,
    ],
]:
    """Get a list of eigenstates for a given operator, assuming it is hermitian."""
    diagonal = eig(operator)
    return Operator(diagonal.basis, diagonal.raw_data)


def eigh_operator[M: BasisMetadata](
    operator: Operator[np.complex128, Basis[StackedMetadata[M, Any], np.complex128]],
) -> Operator[
    np.complex128,
    DiagonalBasis[
        np.complex128,
        ExplicitUnitaryBasis[M, Any],
        ExplicitUnitaryBasis[M, Any],
        Any,
    ],
]:
    """Get a list of eigenstates for a given operator, assuming it is hermitian."""
    diagonal = eigh(operator)
    return Operator(diagonal.basis, diagonal.raw_data)
