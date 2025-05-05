from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from slate_core import Ctype, FundamentalBasis, TupleBasis, TupleMetadata, array, basis
from slate_core.basis import AsUpcast, Basis, DiagonalBasis
from slate_core.linalg import einsum
from slate_core.linalg import into_diagonal as into_diagonal_array
from slate_core.linalg import into_diagonal_hermitian as into_diagonal_hermitian_array
from slate_core.metadata import BasisMetadata

from slate_quantum._util import with_list_basis
from slate_quantum.metadata._label import EigenvalueMetadata
from slate_quantum.operator._operator import (
    Operator,
    OperatorBasis,
    OperatorList,
    OperatorListBasis,
    OperatorMetadata,
)
from slate_quantum.state._basis import EigenstateBasis
from slate_quantum.state._state import StateList

if TYPE_CHECKING:
    from slate_core.explicit_basis import (
        ExplicitDiagonalBasis,
        UpcastExplicitBasisWithMetadata,
    )


def into_diagonal[M: BasisMetadata](
    operator: Operator[OperatorBasis[M], np.dtype[np.complexfloating]],
) -> Operator[
    AsUpcast[
        ExplicitDiagonalBasis[M, M, None, Ctype[np.complexfloating]],
        OperatorMetadata,
        Ctype[np.complexfloating],
    ],
    np.dtype[np.complexfloating],
]:
    """Get a list of eigenstates for a given operator, assuming it is hermitian."""
    diagonal = into_diagonal_array(operator)
    return Operator.build(diagonal.basis.upcast(), diagonal.raw_data).assert_ok()


def into_diagonal_hermitian[M: BasisMetadata, DT: np.dtype[np.complexfloating]](
    operator: Operator[OperatorBasis[M], DT],
) -> Operator[
    AsUpcast[
        ExplicitDiagonalBasis[M, M, None, Ctype[np.complexfloating]],
        OperatorMetadata[M],
        Ctype[np.complexfloating],
    ],
    np.dtype[np.complexfloating],
]:
    """Get a list of eigenstates for a given operator, assuming it is hermitian."""
    diagonal = into_diagonal_hermitian_array(operator)
    inner_basis = diagonal.basis.inner
    transform = inner_basis.children[0].inner.transform()
    data_id = inner_basis.children[0].inner.data_id

    inner: UpcastExplicitBasisWithMetadata[M, Ctype[np.complexfloating]] = (
        EigenstateBasis(transform, direction="forward", data_id=data_id)
        .upcast()
        .resolve_ctype()
    )
    new_basis = (
        DiagonalBasis(
            TupleBasis(
                (
                    inner,
                    EigenstateBasis(transform, direction="forward", data_id=data_id)
                    .upcast()
                    .resolve_ctype(),
                ),
                diagonal.basis.metadata().extra,
            ).resolve_ctype(),
        )
        .resolve_ctype()
        .upcast()
        .resolve_ctype()
    )
    return Operator.build(new_basis, diagonal.raw_data).assert_ok()


def get_eigenstates_hermitian[M: BasisMetadata, DT: np.dtype[np.complexfloating]](
    operator: Operator[OperatorBasis[M], DT],
) -> StateList[
    AsUpcast[
        TupleBasis[tuple[Basis[EigenvalueMetadata], Basis[M]], None],
        TupleMetadata[tuple[EigenvalueMetadata, M], None],
    ]
]:
    diagonal = into_diagonal_hermitian(operator)
    states = diagonal.basis.inner.inner.children[0].inner.eigenvectors().assert_ok()
    as_tuple = with_list_basis(
        states, basis.from_metadata(states.basis.metadata().children[0])
    ).assert_ok()
    out_basis = basis.TupleBasis(
        (
            FundamentalBasis(EigenvalueMetadata(diagonal.raw_data)),
            cast("Basis[M]", as_tuple.basis.inner.children[1]),
        ),
    ).upcast()
    return StateList.build(
        out_basis, as_tuple.raw_data.astype(np.complexfloating)
    ).assert_ok()


def matmul[M0: BasisMetadata](
    lhs: Operator[OperatorBasis[M0], np.dtype[np.complexfloating]],
    rhs: Operator[OperatorBasis[M0], np.dtype[np.complexfloating]],
) -> Operator[OperatorBasis[M0], np.dtype[np.complexfloating]]:
    """
    Multiply each operator in rhs by lhs.

    Aij Bjk = Mik

    Parameters
    ----------
    lhs : Operator[OperatorBasis[M0], np.dtype[np.complexfloating]]
    rhs : Operator[OperatorBasis[M0], np.dtype[np.complexfloating]]

    Returns
    -------
    OperatorList[B_3, B_0, B_2]
    """
    data = einsum("(i k'),(k j) -> (i j)", lhs, rhs)
    return Operator.build(data.basis, data.raw_data).ok()


def commute[M0: BasisMetadata](
    lhs: Operator[OperatorBasis[M0], np.dtype[np.complexfloating]],
    rhs: Operator[OperatorBasis[M0], np.dtype[np.complexfloating]],
) -> Operator[OperatorBasis[M0], np.dtype[np.complexfloating]]:
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
    operator: Operator[OperatorBasis[M0], np.dtype[np.complexfloating]],
) -> Operator[OperatorBasis[M0], np.dtype[np.complexfloating]]:
    """Get the hermitian conjugate of an operator."""
    res = array.dagger(operator)
    # TODO: what should array.dagger's basis be?  # noqa: FIX002
    return Operator.build(res.basis.dual_basis(), res.raw_data).assert_ok()


def dagger_each[M0: BasisMetadata, M1: BasisMetadata](
    operators: OperatorList[
        OperatorListBasis[M0, OperatorMetadata[M1]], np.dtype[np.complexfloating]
    ],
) -> OperatorList[
    OperatorListBasis[M0, OperatorMetadata[M1]], np.dtype[np.complexfloating]
]:
    """Get the hermitian conjugate of an operator."""
    daggered = [dagger(operator) for operator in operators]
    if len(daggered) == 0:
        return OperatorList.build(operators.basis, np.array([])).assert_ok()

    out = OperatorList.from_operators(daggered)
    return OperatorList.build(
        TupleBasis(
            (
                AsUpcast(
                    basis.from_metadata(operators.basis.metadata().children[0]),
                    operators.basis.metadata().children[0],
                ),
                out.basis.inner.children[1],
            )
        ).upcast(),
        out.raw_data,
    ).assert_ok()


def matmul_list_operator[M0: BasisMetadata, M1: OperatorMetadata](
    lhs: OperatorList[OperatorListBasis[M0, M1], np.dtype[np.complexfloating]],
    rhs: Operator[Basis[M1], np.dtype[np.complexfloating]],
) -> OperatorList[OperatorListBasis[M0, M1], np.dtype[np.complexfloating]]:
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
    return OperatorList.build(data.basis, data.raw_data).ok()


def matmul_operator_list[M0: BasisMetadata, M1: OperatorMetadata](
    lhs: Operator[Basis[M1], np.dtype[np.complexfloating]],
    rhs: OperatorList[OperatorListBasis[M0, M1], np.dtype[np.complexfloating]],
) -> OperatorList[OperatorListBasis[M0, M1], np.dtype[np.complexfloating]]:
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
    return OperatorList.build(data.basis, data.raw_data).ok()


def get_commutator_operator_list[M0: BasisMetadata, M1: OperatorMetadata](
    lhs: Operator[Basis[M1], np.dtype[np.complexfloating]],
    rhs: OperatorList[OperatorListBasis[M0, M1], np.dtype[np.complexfloating]],
) -> OperatorList[OperatorListBasis[M0, M1], np.dtype[np.complexfloating]]:
    """
    Given two operators lhs, rhs, calculate the commutator.

    This is equivalent to lhs rhs - rhs lhs.
    """
    # TODO: we want to save on transforming the basis twice, but fast diagonal support  # noqa: FIX002
    # will not play well with this!
    lhs_rhs = matmul_operator_list(lhs, rhs)
    rhs_lhs = matmul_list_operator(rhs, lhs)
    return lhs_rhs - rhs_lhs
