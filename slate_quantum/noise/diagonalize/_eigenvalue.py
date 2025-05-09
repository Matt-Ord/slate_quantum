from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from slate_core import basis as basis_
from slate_core.basis import FundamentalBasis

from slate_quantum.metadata import EigenvalueMetadata
from slate_quantum.noise._kernel import diagonal_kernel_with_outer_basis
from slate_quantum.noise.legacy import tuple_basis
from slate_quantum.operator._operator import OperatorList

if TYPE_CHECKING:
    from slate_core.metadata import BasisMetadata

    from slate_quantum.noise._kernel import (
        DiagonalNoiseKernelWithMetadata,
        NoiseKernelWithMetadata,
    )
    from slate_quantum.operator._operator import OperatorListWithMetadata


def get_periodic_noise_operators_eigenvalue[M: BasisMetadata](
    kernel: NoiseKernelWithMetadata[M, np.dtype[np.complexfloating]],
) -> OperatorListWithMetadata[EigenvalueMetadata, M, np.dtype[np.complexfloating]]:
    r"""
    Given a noise kernel, find the noise operator which diagonalizes the kernel.

    Eigenvalue method.

    Note these are the operators `L`
    """
    converted = kernel.with_basis(basis_.as_tuple(kernel.basis).upcast()).assert_ok()
    converted_second = converted.with_basis(
        tuple_basis(
            (
                basis_.as_tuple(converted.basis.inner.children[0]),
                basis_.as_tuple(converted.basis.inner.children[0]).dual_basis(),
            )
        ).upcast()
    ).assert_ok()
    data = (
        converted_second.raw_data.reshape(
            *converted_second.basis.inner.children[0].shape,
            *converted_second.basis.inner.children[1].shape,
        )
        .swapaxes(0, 1)
        .reshape(converted_second.basis.inner.shape)
    )
    # Find the n^2 operators which are independent
    # I think this is always true
    np.testing.assert_array_almost_equal(data, np.conj(np.transpose(data)))

    res = np.linalg.eigh(data)
    np.testing.assert_array_almost_equal(
        data,
        np.einsum(  # type: ignore unknown
            "ak,k,kb->ab",
            res.eigenvectors,
            res.eigenvalues,
            np.conj(np.transpose(res.eigenvectors)),
        ),
    )
    # The original kernel has the noise operators as \ket{i}\bra{j}
    # When we diagonalize we have \hat{Z}'_\beta = U^\dagger_{\beta, \alpha} \hat{Z}_\alpha
    # np.conj(res.eigenvectors) is U^\dagger_{\beta, \alpha}
    data = np.conj(np.transpose(res.eigenvectors)).reshape(-1)
    eigenvalue = FundamentalBasis(EigenvalueMetadata(res.eigenvalues))

    basis = tuple_basis(
        (eigenvalue, converted_second.basis.inner.children[0].upcast())
    ).upcast()
    return OperatorList.build(basis, data).assert_ok()


def get_periodic_noise_operators_diagonal_eigenvalue[M: BasisMetadata](
    kernel: DiagonalNoiseKernelWithMetadata[M, np.dtype[np.complexfloating]],
) -> OperatorListWithMetadata[EigenvalueMetadata, M, np.dtype[np.complexfloating]]:
    r"""
    For a diagonal kernel it is possible to find N independent noise sources, each of which is diagonal.

    Each of these will be represented by a particular noise operator
    ```latex
    Z_i \ket{i}\bra{i}
    ```
    Note we return a list of noise operators, rather than a single noise operator,
    as it is not currently possible to represent a sparse StackedBasis (unless it can
    be represented as a StackedBasis of individual sparse Basis)
    """
    converted = diagonal_kernel_with_outer_basis(
        kernel, basis_.as_tuple(kernel.basis.inner.outer_recast).upcast()
    )

    data = kernel.raw_data.reshape(converted.basis.inner.outer_recast.inner.shape)  # type: ignore we need to improve the typing of RecastBasis
    # Find the n^2 operators which are independent

    # This should be true if our operators are hermitian - a requirement
    # for our finite temperature correction.
    # For isotropic noise, it is always possible to force this to be true
    # As long as we have evenly spaced k (we have to take symmetric and antisymmetric combinations)
    np.testing.assert_allclose(
        data, np.conj(np.transpose(data)), err_msg="kernel non hermitian"
    )
    res = np.linalg.eigh(data)

    np.testing.assert_allclose(
        data,
        np.einsum(  # type: ignore einsum
            "k,ak,kb->ab",
            res.eigenvalues,
            res.eigenvectors,
            np.conj(np.transpose(res.eigenvectors)),
        ),
        rtol=1e-4,
    )
    # The original kernel has the noise operators as \ket{i}\bra{j}
    # When we diagonalize we have \hat{Z}'_\beta = U^\dagger_{\beta, \alpha} \hat{Z}_\alpha
    # np.conj(res.eigenvectors) is U^\dagger_{\beta, \alpha}

    data = np.conj(np.transpose(res.eigenvectors))
    eigenvalue = FundamentalBasis(EigenvalueMetadata(res.eigenvalues))

    basis = tuple_basis((eigenvalue, converted.basis.inner.outer_recast)).upcast()
    return OperatorList.build(basis, data).assert_ok()
