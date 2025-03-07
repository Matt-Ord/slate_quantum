from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from slate_core import FundamentalBasis, TupleBasis, array, basis, ctype
from slate_core.linalg import einsum
from slate_core.linalg import into_diagonal as into_diagonal_array
from slate_core.linalg import into_diagonal_hermitian as into_diagonal_hermitian_array
from slate_core.metadata import BasisMetadata

from slate_quantum.metadata._label import EigenvalueMetadata
from slate_quantum.operator._operator import (
    Operator,
    OperatorBasis,
    OperatorList,
    OperatorMetadata,
)
from slate_quantum.state._basis import EigenstateBasis
from slate_quantum.state._state import StateList

if TYPE_CHECKING:
    from slate_core.basis import AsUpcast, Basis, DiagonalBasis


def into_diagonal[M: BasisMetadata, DT: ctype[np.complexfloating]](
    operator: Operator[
        OperatorBasis[M, ctype[np.complexfloating]], np.dtype[np.complexfloating]
    ],
) -> Operator[
    AsUpcast[
        DiagonalBasis[
            TupleBasis[tuple[Basis[M, DT], Basis[M, DT]], None],
            ctype[np.complexfloating[Any, Any]],
        ],
        OperatorMetadata,
        ctype[np.complexfloating[Any, Any]],
    ],
    np.dtype[np.complexfloating],
]:
    """Get a list of eigenstates for a given operator, assuming it is hermitian."""
    diagonal = into_diagonal_array(operator)
    return Operator.build(diagonal.basis.upcast(), diagonal.raw_data).ok()


def into_diagonal_hermitian[M: BasisMetadata, DT: np.complexfloating](
    operator: Operator[M, DT],
) -> Operator[
    M,
    np.complexfloating,
    DiagonalBasis[DT, EigenstateBasis[M], EigenstateBasis[M], None],
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
    return Operator(new_basis, diagonal.raw_data)


def get_eigenstates_hermitian[M: BasisMetadata, DT: np.complexfloating](
    operator: Operator[M, DT],
) -> StateList[EigenvalueMetadata, M]:
    diagonal = into_diagonal_hermitian(operator)
    states = diagonal.basis.inner[0].eigenvectors
    as_tuple = states.with_list_basis(basis.from_metadata(states.basis.metadata()[0]))
    out_basis = basis.tuple_basis(
        (FundamentalBasis(EigenvalueMetadata(diagonal.raw_data)), as_tuple.basis[1]),
    )
    return StateList(out_basis, as_tuple.raw_data)


def matmul[M0: BasisMetadata](
    lhs: Operator[M0, np.complexfloating],
    rhs: Operator[M0, np.complexfloating],
) -> Operator[M0, np.complexfloating]:
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
    return Operator(cast("Basis[Any, Any]", data.basis), data.raw_data)


def commute[M0: BasisMetadata](
    lhs: Operator[M0, np.complexfloating],
    rhs: Operator[M0, np.complexfloating],
) -> Operator[M0, np.complexfloating]:
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
    operator: Operator[M0, np.complexfloating],
) -> Operator[M0, np.complexfloating]:
    """Get the hermitian conjugate of an operator."""
    res = array.dagger(operator)
    # TODO: what should array.dagger's basis be?  # noqa: FIX002
    return Operator(res.basis.dual_basis(), res.raw_data)


def dagger_each[M0: BasisMetadata, M1: BasisMetadata](
    operators: OperatorList[M0, M1, np.complexfloating],
) -> OperatorList[M0, M1, np.complexfloating]:
    """Get the hermitian conjugate of an operator."""
    daggered = [dagger(operator) for operator in operators]
    if len(daggered) == 0:
        return OperatorList(operators.basis, np.array([]))

    out = OperatorList.from_operators(daggered)
    return OperatorList(
        tuple_basis((basis.from_metadata(operators.basis.metadata()[0]), out.basis[1])),
        out.raw_data,
    )


def matmul_list_operator[M0: BasisMetadata, M1: BasisMetadata](
    lhs: OperatorList[M0, M1, np.complexfloating],
    rhs: Operator[M1, np.complexfloating],
) -> OperatorList[M0, M1, np.complexfloating]:
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
    return OperatorList(cast("Basis[Any, Any]", data.basis), data.raw_data)


def matmul_operator_list[M0: BasisMetadata, M1: BasisMetadata](
    lhs: Operator[M1, np.complexfloating],
    rhs: OperatorList[M0, M1, np.complexfloating],
) -> OperatorList[M0, M1, np.complexfloating]:
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
    return OperatorList(cast("Basis[Any, Any]", data.basis), data.raw_data)


def get_commutator_operator_list[M0: BasisMetadata, M1: BasisMetadata](
    lhs: Operator[M1, np.complexfloating],
    rhs: OperatorList[M0, M1, np.complexfloating],
) -> OperatorList[M0, M1, np.complexfloating]:
    """
    Given two operators lhs, rhs, calculate the commutator.

    This is equivalent to lhs rhs - rhs lhs.
    """
    # TODO: we want to save on transforming the basis twice, but fast diagonal support  # noqa: FIX002
    # will not play well with this!
    lhs_rhs = matmul_operator_list(lhs, rhs)
    rhs_lhs = matmul_list_operator(rhs, lhs)
    return lhs_rhs - rhs_lhs
