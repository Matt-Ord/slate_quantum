from __future__ import annotations

from typing import TYPE_CHECKING, Any

from slate.basis.stacked import diagonal_basis
from slate.linalg import eig, eigh
from slate.metadata import BasisMetadata, StackedMetadata

from slate_quantum.model.operator._operator import Operator
from slate_quantum.model.state.eigenstate_basis import EigenstateBasis

if TYPE_CHECKING:
    import numpy as np
    from slate.basis import Basis, DiagonalBasis
    from slate.explicit_basis import ExplicitBasis


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
        EigenstateBasis[M],
        EigenstateBasis[M],
        Any,
    ],
]:
    """Get a list of eigenstates for a given operator, assuming it is hermitian."""
    diagonal = eigh(operator)
    inner_basis = diagonal.basis.inner[0]
    new_inner_basis = EigenstateBasis[M](
        inner_basis._data,  # noqa: SLF001
        direction=inner_basis.direction,
        data_id=inner_basis._data_id,  # noqa: SLF001
    )
    new_basis = diagonal_basis(
        (new_inner_basis, new_inner_basis.conjugate_basis()),
        diagonal.basis.metadata.extra,
    )
    return Operator(new_basis, diagonal.raw_data)
