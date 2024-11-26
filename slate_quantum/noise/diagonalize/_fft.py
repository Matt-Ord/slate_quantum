from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from slate.basis import (
    Basis,
    DiagonalBasis,
    FundamentalBasis,
    TupleBasis,
    TupleBasis2D,
    as_tuple_basis,
    diagonal_basis,
    fundamental_basis_from_shape,
    tuple_basis,
)
from slate.metadata import (
    SimpleMetadata,
    StackedMetadata,
    fundamental_stacked_nk_points,
)
from slate.metadata.util import fundamental_nk_points
from slate.util import pad_ft_points

from slate_quantum.model import EigenvalueMetadata
from slate_quantum.model.operator import OperatorList
from slate_quantum.noise.kernel import (
    as_axis_kernel_from_isotropic,
    get_diagonal_noise_operators_from_axis,
)

if TYPE_CHECKING:
    from slate.metadata import BasisMetadata, Metadata2D

    from slate_quantum.noise.kernel import IsotropicNoiseKernel


def _assert_periodic_sample(
    basis_shape: tuple[int, ...], shape: tuple[int, ...]
) -> None:
    ratio = tuple(n % s for n, s in zip(basis_shape, shape, strict=True))
    # Is 2 * np.pi * N / s equal to A * 2 * np.pi for some integer A
    message = (
        "Operators requested for a sample which does not evenly divide the basis shape\n"
        "This would result in noise operators which are not periodic"
    )
    try:
        np.testing.assert_array_almost_equal(ratio, 0, err_msg=message)
    except AssertionError:
        raise AssertionError(message) from None


def _get_operators_for_isotropic_noise[M: BasisMetadata](
    basis: Basis[M, Any],
    *,
    n: int | None = None,
    fundamental_n: int | None = None,
    assert_periodic: bool = True,
) -> OperatorList[
    Metadata2D[SimpleMetadata, Metadata2D[M, M, None], None],
    np.complex128,
    TupleBasis2D[
        np.complex128,
        FundamentalBasis[SimpleMetadata],
        DiagonalBasis[
            np.complex128, Basis[M, np.complex128], Basis[M, np.complex128], None
        ],
        None,
    ],
]:
    fundamental_n = basis.size if fundamental_n is None else fundamental_n
    if assert_periodic:
        _assert_periodic_sample((basis.size,), (fundamental_n,))
    n = fundamental_n if n is None else n
    # Operators e^(ik_n x_m) / sqrt(M)
    # with k_n = 2 * np.pi * n / N, n = 0...N
    # and x_m = m, m = 0...M
    k = 2 * np.pi / fundamental_n
    nk_points = fundamental_nk_points(SimpleMetadata(basis.size))[np.newaxis, :]
    i_points = np.arange(0, n)[:, np.newaxis]

    operators = np.exp(1j * i_points * k * nk_points) / np.sqrt(basis.size)
    return OperatorList(
        tuple_basis((FundamentalBasis.from_size(n), diagonal_basis((basis, basis)))),
        operators,
    )


def _get_noise_eigenvalues_isotropic_fft[M: BasisMetadata](
    kernel: IsotropicNoiseKernel[M, np.complex128], *, fundamental_n: int | None = None
) -> EigenvalueMetadata:
    fundamental_n = kernel.basis.size if fundamental_n is None else fundamental_n
    coefficients = np.fft.ifft(
        pad_ft_points(kernel.raw_data, (fundamental_n,), (0,)),
        norm="forward",
    )
    coefficients *= kernel.basis.size / fundamental_n
    return EigenvalueMetadata(coefficients)


def get_periodic_noise_operators_isotropic_fft[M: BasisMetadata](
    kernel: IsotropicNoiseKernel[M, np.complex128],
    *,
    fundamental_n: int | None = None,
    assert_periodic: bool = True,
) -> OperatorList[
    Metadata2D[EigenvalueMetadata, Metadata2D[M, M, None], None],
    np.complex128,
    TupleBasis2D[
        np.complex128,
        FundamentalBasis[EigenvalueMetadata],
        DiagonalBasis[
            np.complex128,
            Basis[M, np.complex128],
            Basis[M, np.complex128],
            None,
        ],
        None,
    ],
]:
    r"""
    For an isotropic noise kernel, the noise operators are independent in k space.

    beta(x - x') = 1 / N \sum_k |f(k)|^2 e^(ikx) for some f(k)
    |f(k)|^2 = \sum_x beta(x) e^(-ik.x)

    The independent noise operators are then given by

    L(k) = 1 / N \sum_x e^(ikx) S(x)

    The inddependent operators can therefore be found directly using a FFT
    of the noise beta(x).


    """
    fundamental_n = kernel.basis.size if fundamental_n is None else fundamental_n

    operators = _get_operators_for_isotropic_noise(
        kernel.basis.inner_recast,
        fundamental_n=fundamental_n,
        assert_periodic=assert_periodic,
    )
    eigenvalues = _get_noise_eigenvalues_isotropic_fft(
        kernel, fundamental_n=fundamental_n
    )
    return OperatorList(
        tuple_basis((FundamentalBasis(eigenvalues), operators.basis.children[1])),
        operators.raw_data,
    )


def get_periodic_operators_for_real_isotropic_noise[B: Basis[BasisMetadata, Any]](
    basis: B,
    *,
    n: int | None = None,
    fundamental_n: int | None = None,
    assert_periodic: bool = True,
) -> OperatorList[
    Metadata2D[SimpleMetadata, Metadata2D[BasisMetadata, BasisMetadata, None], None],
    np.complex128,
    TupleBasis2D[
        np.complex128,
        FundamentalBasis[SimpleMetadata],
        DiagonalBasis[
            np.complex128,
            B,
            B,
            None,
        ],
        None,
    ],
]:
    """Get operators used for real isotropic noise.

    returns the 2n - 1 smallest operators (n frequencies)


    """
    fundamental_n = basis.size if fundamental_n is None else fundamental_n
    if assert_periodic:
        _assert_periodic_sample((basis.size,), (fundamental_n,))
    n = (fundamental_n // 2) if n is None else n

    k = 2 * np.pi / fundamental_n
    nk_points = fundamental_nk_points(SimpleMetadata(basis.size))[np.newaxis, :]

    sines = np.sin(np.arange(n - 1, 0, -1)[:, np.newaxis] * nk_points * k)
    coses = np.cos(np.arange(0, n)[:, np.newaxis] * nk_points * k)
    data = np.concatenate([coses, sines]).astype(np.complex128) / np.sqrt(basis.size)

    # Equivalent to
    # ! data = standard_operators["data"].reshape(kernel["basis"].n, -1)
    # ! end = fundamental_n // 2 + 1
    # Build (e^(ikx) +- e^(-ikx)) operators
    # ! data[1:end] = np.sqrt(2) * np.real(data[1:end])
    # ! data[end:] = np.sqrt(2) * np.imag(np.conj(data[end:]))
    return OperatorList(
        tuple_basis((FundamentalBasis.from_size(n), diagonal_basis((basis, basis)))),
        data,
    )


def get_periodic_noise_operators_real_isotropic_fft[M: BasisMetadata](
    kernel: IsotropicNoiseKernel[M, np.complex128],
    *,
    n: int | None = None,
    assert_periodic: bool = True,
) -> OperatorList[
    Metadata2D[EigenvalueMetadata, Metadata2D[M, M, None], None],
    np.complex128,
    TupleBasis2D[
        np.complex128,
        FundamentalBasis[EigenvalueMetadata],
        DiagonalBasis[
            np.complex128,
            Basis[StackedMetadata[M, None], np.complex128],
            Basis[StackedMetadata[M, None], np.complex128],
            None,
        ],
        None,
    ],
]:
    r"""
    For an isotropic noise kernel, the noise operators are independent in k space.

    beta(x - x') = 1 / N \sum_k |f(k)|^2 e^(ikx) for some f(k)
    |f(k)|^2 = \sum_x beta(x) e^(-ik.x)

    The independent noise operators are then given by

    L(k) = 1 / N \sum_x e^(ikx) S(x)

    The inddependent operators can therefore be found directly using a FFT
    of the noise beta(x).

    For a real kernel, the coefficients of  e^(+-ikx) are the same
    we can therefore equivalently use cos(x) and sin(x) as the basis
    for the kernel.


    """
    np.testing.assert_allclose(np.imag(kernel.raw_data), 0)

    fundamental_n = kernel.basis.size if n is None else 2 * n + 1

    operators = get_periodic_operators_for_real_isotropic_noise(
        kernel.basis, fundamental_n=fundamental_n, assert_periodic=assert_periodic
    )

    eigenvalues = _get_noise_eigenvalues_isotropic_fft(
        kernel, fundamental_n=fundamental_n
    )

    np.testing.assert_allclose(
        eigenvalues.values[1::],
        eigenvalues.values[1::][::-1],
        rtol=1e-8,
    )

    return OperatorList(
        tuple_basis((FundamentalBasis(eigenvalues), operators.basis.children[1])),
        operators.raw_data,
    )


def _get_operators_for_isotropic_stacked_noise[M: StackedMetadata[BasisMetadata, None]](
    basis: Basis[M, Any],
    *,
    shape: tuple[int, ...] | None = None,
    fundamental_shape: tuple[int, ...] | None = None,
    assert_periodic: bool = True,
) -> OperatorList[
    Metadata2D[SimpleMetadata, Metadata2D[M, M, None], None],
    np.complex128,
    TupleBasis2D[
        np.complex128,
        TupleBasis[BasisMetadata, None, np.generic],
        DiagonalBasis[
            np.complex128,
            Basis[M, np.generic],
            Basis[M, np.generic],
            None,
        ],
        None,
    ],
]:
    converted_basis = as_tuple_basis(basis)
    fundamental_shape = (
        converted_basis.shape if fundamental_shape is None else fundamental_shape
    )
    if assert_periodic:
        _assert_periodic_sample(converted_basis.shape, fundamental_shape)
    shape = fundamental_shape if shape is None else shape
    # Operators e^(ik_n0,n1,.. x_m0,m1,..) / sqrt(prod(Mi))
    # with k_n0,n1 = 2 * np.pi * (n0,n1,...) / prod(Ni), ni = 0...Ni
    # and x_m0,m1 = (m0,m1,...), mi = 0...Mi
    k = tuple(2 * np.pi / n for n in fundamental_shape)
    shape_basis = fundamental_basis_from_shape(fundamental_shape)

    nk_points = fundamental_stacked_nk_points(basis.metadata())
    i_points = fundamental_stacked_nk_points(
        StackedMetadata.from_shape(converted_basis.shape, extra=None)
    )

    operators = np.array(
        [
            np.exp(1j * np.einsum("i,i,ij->j", k, i, nk_points))  # type: ignore einsum
            / np.sqrt(converted_basis.size)
            for i in zip(*i_points)
        ]
    )
    converted_basis = cast(Basis[M, Any], converted_basis)
    return OperatorList(
        tuple_basis(
            (
                shape_basis,
                diagonal_basis((converted_basis, converted_basis.dual_basis())),
            )
        ),
        operators,
    )


def _get_noise_eigenvalues_isotropic_stacked_fft[
    M: StackedMetadata[BasisMetadata, Any]
](
    kernel: IsotropicNoiseKernel[M, np.complex128],
    *,
    fundamental_shape: tuple[int, ...] | None = None,
) -> EigenvalueMetadata:
    converted_outer = as_tuple_basis(kernel.basis.outer_recast)
    converted = kernel.with_isotropic_basis(converted_outer)
    fundamental_shape = (
        converted_outer.shape if fundamental_shape is None else fundamental_shape
    )

    coefficients = np.fft.ifftn(
        pad_ft_points(
            converted.raw_data.reshape(converted_outer.shape),
            fundamental_shape,
            tuple(range(len(fundamental_shape))),
        ),
        norm="forward",
    )

    coefficients *= converted_outer.size / coefficients.size
    return EigenvalueMetadata(coefficients)


def get_periodic_noise_operators_isotropic_stacked_fft[
    M: StackedMetadata[BasisMetadata, Any]
](
    kernel: IsotropicNoiseKernel[M, np.complex128],
    *,
    fundamental_shape: tuple[int, ...] | None = None,
    assert_periodic: bool = True,
) -> OperatorList[
    Metadata2D[EigenvalueMetadata, Metadata2D[M, M, None], None],
    np.complex128,
    TupleBasis2D[
        np.complex128,
        FundamentalBasis[EigenvalueMetadata],
        DiagonalBasis[
            np.complex128,
            Basis[M, Any],
            Basis[M, Any],
            None,
        ],
        None,
    ],
]:
    r"""
    For an isotropic noise kernel, the noise operators are independent in k space.

    beta(x - x') = 1 / N \sum_k |f(k)|^2 e^(ikx) for some f(k)
    |f(k)|^2 = \sum_x beta(x) e^(-ik.x)

    The independent noise operators are then given by

    L(k) = 1 / N \sum_x e^(ikx) S(x)

    The inddependent operators can therefore be found directly using a FFT
    of the noise beta(x).
    """
    converted_outer = as_tuple_basis(kernel.basis.outer_recast)
    converted = kernel.with_isotropic_basis(converted_outer)
    fundamental_shape = (
        converted_outer.shape if fundamental_shape is None else fundamental_shape
    )

    operators = _get_operators_for_isotropic_stacked_noise(
        converted.basis.outer_recast,
        fundamental_shape=fundamental_shape,
        assert_periodic=assert_periodic,
    )
    eigenvalues = _get_noise_eigenvalues_isotropic_stacked_fft(
        converted, fundamental_shape=fundamental_shape
    )
    return OperatorList(
        tuple_basis((FundamentalBasis(eigenvalues), operators.basis.children[1])),
        operators.raw_data,
    )


def get_periodic_noise_operators_real_isotropic_stacked_fft[
    M: StackedMetadata[BasisMetadata, Any]
](
    kernel: IsotropicNoiseKernel[M, np.complex128],
    *,
    fundamental_shape: tuple[int | None, ...] | None = None,
    assert_periodic: bool = True,
) -> OperatorList[
    Metadata2D[EigenvalueMetadata, Metadata2D[M, M, None], None],
    np.complex128,
    TupleBasis2D[
        Any,
        FundamentalBasis[EigenvalueMetadata],
        Basis[Metadata2D[M, M, None], Any],
        Any,
    ],
]:
    """Calculate the noise operators for a general isotropic noise kernel.

    Polynomial fitting to get Taylor expansion.

    Parameters
    ----------
    kernel: IsotropicNoiseKernel[TupleBasisWithLengthLike[Any, Any]]
    n: int, by default 1

    Returns
    -------
    The noise operators formed using the 2n+1 lowest fourier terms, and the corresponding coefficients.

    """
    axis_kernels = as_axis_kernel_from_isotropic(kernel)

    operators = tuple(
        get_periodic_noise_operators_isotropic_fft(
            kernel,
            fundamental_n=None if (fundamental_shape is None) else fundamental_shape[i],
            assert_periodic=assert_periodic,
        )
        for (i, kernel) in enumerate(axis_kernels)
    )

    return get_diagonal_noise_operators_from_axis(  # type: ignore cast here is safe
        operators, kernel.basis.outer_recast.metadata().extra
    )
