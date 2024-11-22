from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from slate.basis import (
    Basis,
    diagonal_basis,
)
from slate.linalg import einsum
from slate.linalg import into_diagonal as into_diagonal_array
from slate.linalg import into_diagonal_hermitian as into_diagonal_hermitian_array
from slate.metadata import BasisMetadata, Metadata2D

from slate_quantum.model.operator import Operator, OperatorList
from slate_quantum.model.state.eigenstate_basis import EigenstateBasis

if TYPE_CHECKING:
    from slate.basis import DiagonalBasis
    from slate.explicit_basis import ExplicitBasis


def into_diagonal[M: BasisMetadata, E, DT: np.complexfloating[Any, Any]](
    operator: Operator[Metadata2D[M, M, E], DT],
) -> Operator[
    Metadata2D[M, M, E],
    np.complex128,
    DiagonalBasis[
        DT,
        ExplicitBasis[M, DT],
        ExplicitBasis[M, DT],
        E,
    ],
]:
    """Get a list of eigenstates for a given operator, assuming it is hermitian."""
    diagonal = into_diagonal_array(operator)
    return Operator(diagonal.basis, diagonal.raw_data)


def into_diagonal_hermitian[M: BasisMetadata, E, DT: np.complexfloating[Any, Any]](
    operator: Operator[Metadata2D[M, M, E], DT],
) -> Operator[
    Metadata2D[M, M, E],
    np.complex128,
    DiagonalBasis[
        DT,
        EigenstateBasis[M],
        EigenstateBasis[M],
        E,
    ],
]:
    """Get a list of eigenstates for a given operator, assuming it is hermitian."""
    diagonal = into_diagonal_hermitian_array(operator)
    inner_basis = diagonal.basis.inner[0]
    # TODO: this doesn't play well with fast diagonal support  # noqa: FIX002
    # Need to use einsum inside ExplicitBasis to prevent conversion of states
    # to a dense array.
    new_inner_basis = EigenstateBasis(
        inner_basis.states,
        direction=inner_basis.direction,
        data_id=inner_basis.data_id,
    )
    new_basis = diagonal_basis(
        (new_inner_basis, new_inner_basis.conjugate_basis()),
        diagonal.basis.metadata().extra,
    )
    return Operator(new_basis, diagonal.raw_data)


def matmul_list_operator[M: Metadata2D[BasisMetadata, BasisMetadata, Any]](
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


def matmul_operator_list[M: Metadata2D[BasisMetadata, BasisMetadata, Any]](
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


def get_commutator_operator_list[M: Metadata2D[BasisMetadata, BasisMetadata, Any]](
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
