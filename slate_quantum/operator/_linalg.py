from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from slate_core import FundamentalBasis, array, basis
from slate_core.linalg import einsum
from slate_core.linalg import into_diagonal as into_diagonal_array
from slate_core.linalg import into_diagonal_hermitian as into_diagonal_hermitian_array
from slate_core.metadata import BasisMetadata

from slate_quantum._util.legacy import diagonal_basis, tuple_basis
from slate_quantum.metadata._label import EigenvalueMetadata
from slate_quantum.operator._operator import (
    LegacyOperator,
    LegacyOperatorList,
    OperatorList,
    build_legacy_operator,
    build_legacy_operator_list,
)
from slate_quantum.state._basis import EigenstateBasis
from slate_quantum.state._state import LegacyStateList, build_legacy_state_list

if TYPE_CHECKING:
    from slate_core.basis import (
        Basis,
    )
    from slate_core.explicit_basis import ExplicitBasis

    from slate_quantum._util.legacy import LegacyDiagonalBasis


def into_diagonal[M: BasisMetadata, DT: np.complexfloating](
    operator: LegacyOperator[M, DT],
) -> LegacyOperator[
    M,
    np.complexfloating,
    LegacyDiagonalBasis[DT, ExplicitBasis[M, DT], ExplicitBasis[M, DT], None],
]:
    """Get a list of eigenstates for a given operator, assuming it is hermitian."""
    diagonal = into_diagonal_array(operator)
    return build_legacy_operator(diagonal.basis, diagonal.raw_data)


def into_diagonal_hermitian[M: BasisMetadata, DT: np.complexfloating](
    operator: LegacyOperator[M, DT],
) -> LegacyOperator[
    M,
    np.complexfloating,
    LegacyDiagonalBasis[DT, EigenstateBasis[M], EigenstateBasis[M], None],
]:
    """Get a list of eigenstates for a given operator, assuming it is hermitian."""
    diagonal = into_diagonal_hermitian_array(operator)
    inner_basis = diagonal.basis.inner

    new_basis = diagonal_basis(
        (
            EigenstateBasis(
                inner_basis[0].matrix,
                direction=inner_basis[0].direction,
                data_id=inner_basis[0].data_id,
            ),
            EigenstateBasis(
                inner_basis[1].matrix,
                direction=inner_basis[1].direction,
                data_id=inner_basis[1].data_id,
            ),
        ),
        diagonal.basis.metadata().extra,
    )
    return build_legacy_operator(new_basis, diagonal.raw_data)


def get_eigenstates_hermitian[M: BasisMetadata, DT: np.complexfloating](
    operator: LegacyOperator[M, DT],
) -> LegacyStateList[EigenvalueMetadata, M]:
    diagonal = into_diagonal_hermitian(operator)
    states = diagonal.basis.inner[0].eigenvectors
    as_tuple = states.with_list_basis(basis.from_metadata(states.basis.metadata()[0]))
    out_basis = tuple_basis(
        (FundamentalBasis(EigenvalueMetadata(diagonal.raw_data)), as_tuple.basis[1]),
    )
    return build_legacy_state_list(out_basis, as_tuple.raw_data)


def matmul[M0: BasisMetadata](
    lhs: LegacyOperator[M0, np.complexfloating],
    rhs: LegacyOperator[M0, np.complexfloating],
) -> LegacyOperator[M0, np.complexfloating]:
    """
    Multiply each operator in rhs by lhs.

    Aij Bjk = Mik

    Parameters
    ----------
    lhs : OperatorList[B_3, B_0, B_1]
    rhs : Operator[B_1, B_2]

    Returns
    -------
    OperatorList[B_3, B_0, B_2]
    """
    data = einsum("(i k'),(k j) -> (i j)", lhs, rhs)
    return build_legacy_operator(cast("Basis[Any, Any]", data.basis), data.raw_data)


def commute[M0: BasisMetadata](
    lhs: LegacyOperator[M0, np.complexfloating],
    rhs: LegacyOperator[M0, np.complexfloating],
) -> LegacyOperator[M0, np.complexfloating]:
    """
    Given two operators lhs, rhs, calculate the commutator.

    This is equivalent to lhs rhs - rhs lhs.
    """
    # TODO: we want to save on transforming the basis twice, but fast diagonal support  # noqa: FIX002
    # will not play well with this!
    lhs_rhs = matmul(lhs, rhs)
    rhs_lhs = matmul(rhs, lhs)
    return lhs_rhs - rhs_lhs


def dagger[M0: BasisMetadata](
    operator: LegacyOperator[M0, np.complexfloating],
) -> LegacyOperator[M0, np.complexfloating]:
    """Get the hermitian conjugate of an operator."""
    res = array.dagger(operator)
    # TODO: what should array.dagger's basis be?  # noqa: FIX002
    return build_legacy_operator(res.basis.dual_basis(), res.raw_data)


def dagger_each[M0: BasisMetadata, M1: BasisMetadata](
    operators: LegacyOperatorList[M0, M1, np.complexfloating],
) -> LegacyOperatorList[M0, M1, np.complexfloating]:
    """Get the hermitian conjugate of an operator."""
    daggered = [dagger(operator) for operator in operators]
    if len(daggered) == 0:
        return build_legacy_operator_list(operators.basis, np.array([]))

    out = OperatorList.from_operators(daggered)
    return build_legacy_operator_list(
        tuple_basis((basis.from_metadata(operators.basis.metadata()[0]), out.basis[1])),
        out.raw_data,
    )


def matmul_list_operator[M0: BasisMetadata, M1: BasisMetadata](
    lhs: LegacyOperatorList[M0, M1, np.complexfloating],
    rhs: LegacyOperator[M1, np.complexfloating],
) -> LegacyOperatorList[M0, M1, np.complexfloating]:
    """
    Multiply each operator in rhs by lhs.

    Aij Bjk = Mik

    Parameters
    ----------
    lhs : OperatorList[B_3, B_0, B_1]
    rhs : Operator[B_1, B_2]

    Returns
    -------
    OperatorList[B_3, B_0, B_2]
    """
    data = einsum("(m (i k')),(k j) -> (m (i j))", lhs, rhs)
    return build_legacy_operator_list(
        cast("Basis[Any, Any]", data.basis), data.raw_data
    )


def matmul_operator_list[M0: BasisMetadata, M1: BasisMetadata](
    lhs: LegacyOperator[M1, np.complexfloating],
    rhs: LegacyOperatorList[M0, M1, np.complexfloating],
) -> LegacyOperatorList[M0, M1, np.complexfloating]:
    """
    Multiply each operator in rhs by lhs.

    Aij Bjk = Mik

    Parameters
    ----------
    lhs : Operator[B_0, B_1]
    rhs : OperatorList[B_3, B_1, B_2]

    Returns
    -------
    OperatorList[B_3, B_0, B_2]
    """
    data = einsum("(i k'),(m (k j)) -> (m (i j))", lhs, rhs)
    return build_legacy_operator_list(
        cast("Basis[Any, Any]", data.basis), data.raw_data
    )


def get_commutator_operator_list[M0: BasisMetadata, M1: BasisMetadata](
    lhs: LegacyOperator[M1, np.complexfloating],
    rhs: LegacyOperatorList[M0, M1, np.complexfloating],
) -> LegacyOperatorList[M0, M1, np.complexfloating]:
    """
    Given two operators lhs, rhs, calculate the commutator.

    This is equivalent to lhs rhs - rhs lhs.
    """
    # TODO: we want to save on transforming the basis twice, but fast diagonal support  # noqa: FIX002
    # will not play well with this!
    lhs_rhs = matmul_operator_list(lhs, rhs)
    rhs_lhs = matmul_list_operator(rhs, lhs)
    return lhs_rhs - rhs_lhs
