from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from slate.basis import (
    Basis,
)
from slate.basis.stacked import (
    diagonal_basis,
)
from slate.linalg import eig, eigh, einsum
from slate.metadata import BasisMetadata, StackedMetadata

from slate_quantum.model.operator import Operator, OperatorList
from slate_quantum.model.state.eigenstate_basis import EigenstateBasis

if TYPE_CHECKING:
    from slate.basis import DiagonalBasis
    from slate.explicit_basis import ExplicitBasis


def eig_operator[M: BasisMetadata, E, DT: np.complexfloating[Any, Any]](
    operator: Operator[StackedMetadata[M, E], DT],
) -> Operator[
    StackedMetadata[M, E],
    np.complex128,
    DiagonalBasis[
        DT,
        ExplicitBasis[M, DT],
        ExplicitBasis[M, DT],
        E,
    ],
]:
    """Get a list of eigenstates for a given operator, assuming it is hermitian."""
    diagonal = eig(operator)
    return Operator(diagonal.basis, diagonal.raw_data)


def eigh_operator[M: BasisMetadata, E, DT: np.complexfloating[Any, Any]](
    operator: Operator[StackedMetadata[M, E], DT],
) -> Operator[
    StackedMetadata[M, E],
    np.complex128,
    DiagonalBasis[
        DT,
        EigenstateBasis[M],
        EigenstateBasis[M],
        E,
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


def matmul_list_operator[M: StackedMetadata[Any, Any]](
    lhs: OperatorList[M, np.complex128],
    rhs: Operator[Any, np.complex128],
) -> OperatorList[M, np.complex128]:
    """
    Multiply each operator in rhs by lhs.

    Aij Bjk = Mik

    Parameters
    ----------
    lhs : OperatorList[_B3, _B0, _B1]
    rhs : Operator[_B1, _B2]

    Returns
    -------
    OperatorList[_B3, _B0, _B2]
    """
    data = einsum("m(ik),kj->m(ij)", lhs, rhs)
    return OperatorList(cast(Basis[M, Any], data.basis), data.raw_data)


def matmul_operator_list[M: StackedMetadata[BasisMetadata, Any]](
    lhs: Operator[Any, np.complex128],
    rhs: OperatorList[M, np.complex128],
) -> OperatorList[M, np.complex128]:
    """
    Multiply each operator in rhs by lhs.

    Aij Bjk = Mik

    Parameters
    ----------
    lhs : Operator[_B0, _B1]
    rhs : OperatorList[_B3, _B1, _B2]

    Returns
    -------
    OperatorList[_B3, _B0, _B2]
    """
    data = einsum("ik,m(kj)->m(ij)", lhs, rhs)
    return OperatorList(cast(Basis[M, Any], data.basis), data.raw_data)


def get_commutator_operator_list[M: StackedMetadata[BasisMetadata, Any]](
    lhs: Operator[Any, np.complex128],
    rhs: OperatorList[M, np.complex128],
) -> OperatorList[M, np.complex128]:
    """
    Given two operators lhs, rhs, calculate the commutator.

    This is equivalent to lhs rhs - rhs lhs.
    """
    # TODO: fast diagonal support  # noqa: FIX002
    # will not play well with this!
    converted = cast(
        OperatorList[M, np.complex128, Basis[M, np.complex128]],
        rhs.with_operator_basis(lhs.basis),
    )
    lhs_rhs = matmul_operator_list(lhs, converted)
    rhs_lhs = matmul_list_operator(converted, lhs)
    return lhs_rhs - rhs_lhs
