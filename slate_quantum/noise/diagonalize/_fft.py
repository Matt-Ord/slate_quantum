from __future__ import annotations

from typing import TYPE_CHECKING, Any, Never

import numpy as np
from slate_core import Basis, Ctype, FundamentalBasis, TupleBasis, TupleMetadata
from slate_core import basis as _basis
from slate_core.util import pad_ft_points

from slate_quantum import operator
from slate_quantum.metadata import EigenvalueMetadata
from slate_quantum.noise._kernel import with_isotropic_basis
from slate_quantum.operator import OperatorList

if TYPE_CHECKING:
    from slate_core.metadata import BasisMetadata

    from slate_quantum.noise._kernel import IsotropicKernelWithMetadata
    from slate_quantum.operator._diagonal import DiagonalOperatorList
    from slate_quantum.operator._operator import (
        OperatorListBasis,
        OperatorMetadata,
    )


def _get_noise_eigenvalues_isotropic_fft[M: BasisMetadata](
    kernel: IsotropicKernelWithMetadata[M, Ctype[Never], np.dtype[np.complexfloating]],
    *,
    fundamental_n: int | None = None,
) -> EigenvalueMetadata:
    fundamental_n = kernel.basis.size if fundamental_n is None else fundamental_n
    coefficients = np.fft.ifft(
        pad_ft_points(kernel.raw_data, (fundamental_n,), (0,)),
        norm="forward",
    )
    coefficients *= kernel.basis.size / fundamental_n
    return EigenvalueMetadata(coefficients)


def get_periodic_noise_operators_isotropic_fft[M: BasisMetadata](
    kernel: IsotropicKernelWithMetadata[M, Ctype[Never], np.dtype[np.complexfloating]],
) -> OperatorList[
    OperatorListBasis[EigenvalueMetadata, OperatorMetadata[M]],
    np.dtype[np.complexfloating],
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
    operators = operator.build.all_axis_periodic_operators(
        kernel.basis.inner.outer_recast.outer_recast,
    )
    operators = operators.with_basis(
        _basis.as_tuple(operators.basis).upcast()
    ).assert_ok()
    eigenvalues = _get_noise_eigenvalues_isotropic_fft(kernel)
    return OperatorList.build(
        TupleBasis(
            (FundamentalBasis(eigenvalues), operators.basis.inner.children[1])
        ).upcast(),
        operators.raw_data,
    ).assert_ok()


def _get_noise_eigenvalues_isotropic_stacked_fft[M: BasisMetadata, E](
    kernel: IsotropicKernelWithMetadata[
        TupleMetadata[tuple[M, ...], E], Ctype[Never], np.dtype[np.complexfloating]
    ],
    *,
    fundamental_shape: tuple[int, ...] | None = None,
) -> EigenvalueMetadata:
    converted_outer = _basis.as_tuple(kernel.basis.inner.outer_recast.outer_recast)
    converted = with_isotropic_basis(kernel, converted_outer)
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


def build_periodic_noise_operators[
    M0: EigenvalueMetadata,
    M: BasisMetadata,
    E,
](
    list_basis: Basis[M0],
    inner_basis: TupleBasis[tuple[Basis[M], ...], E, Any],
) -> DiagonalOperatorList[
    M0, TupleMetadata[tuple[M, ...], E], np.dtype[np.complexfloating]
]:
    r"""
    For an isotropic noise kernel, the noise operators are independent in k space.

    .. math::
        \beta(x - x') = \frac{1}{N} \sum_k |f(k)|^2 e^{ikx} for some f(k)
        |f(k)|^2 = \sum_x \beta(x) e^{-ik.x}

    The independent noise operators are then given by

    .. math::
        \hat{f}(k) = \sqrt{N} f(k)

    Parameters
    ----------
    kernel : IsotropicNoiseKernel[M, np.complexfloating]

    Returns
    -------
    OperatorList[
        EigenvalueMetadata,
        M,
        np.complexfloating,
        TupleBasis2D[
            Any,
            FundamentalBasis[SimpleMetadata],
            RecastDiagonalOperatorBasis[M, Any],
            None,
        ],
    ]
    """
    operators = operator.build.all_periodic_operators(inner_basis)
    operators = operators.with_basis(operators.basis.inner.inner.upcast()).assert_ok()
    operators.basis.inner.children[1]

    return OperatorList.build(
        TupleBasis(
            (FundamentalBasis(list_basis.metadata()), operators.basis.inner.children[1])
        ).upcast(),
        operators.raw_data,
    ).assert_ok()


def get_periodic_noise_operators_isotropic_stacked_fft[M: BasisMetadata, E](
    kernel: IsotropicKernelWithMetadata[
        TupleMetadata[tuple[M, ...], E], Ctype[Never], np.dtype[np.complexfloating]
    ],
    *,
    fundamental_shape: tuple[int, ...] | None = None,
) -> DiagonalOperatorList[
    EigenvalueMetadata, TupleMetadata[tuple[M, ...], E], np.dtype[np.complexfloating]
]:
    r"""
    For an isotropic noise kernel, the noise operators are independent in k space.

    beta(x - x') = 1 / N \sum_k |f(k)|^2 e^(ikx) for some f(k)
    |f(k)|^2 = \sum_x beta(x) e^(-ik.x)

    The independent noise operators are then given by

    L(k) = 1 / N \sum_x e^(ikx) S(x)

    The independent operators can therefore be found directly using a FFT
    of the noise beta(x).
    """
    converted_outer = _basis.as_tuple(kernel.basis.inner.outer_recast.inner_recast)
    converted = with_isotropic_basis(kernel, converted_outer)
    fundamental_shape = converted_outer.shape

    eigenvalues = _get_noise_eigenvalues_isotropic_stacked_fft(
        converted, fundamental_shape=fundamental_shape
    )
    return build_periodic_noise_operators(
        FundamentalBasis(eigenvalues), converted_outer
    )
