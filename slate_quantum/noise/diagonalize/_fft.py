from __future__ import annotations

from typing import TYPE_CHECKING, Any, Never

import numpy as np
from slate_core import Basis, Ctype, TupleBasis, TupleMetadata
from slate_core.basis import (
    FundamentalBasis,
    TupleBasis,
    as_tuple_basis,
)
from slate_core.util import pad_ft_points

from slate_quantum import operator
from slate_quantum.metadata import EigenvalueMetadata
from slate_quantum.operator import OperatorList

if TYPE_CHECKING:
    from slate_core.metadata import BasisMetadata

    from slate_quantum.noise._kernel import IsotropicKernelWithMetadata
    from slate_quantum.operator._operator import (
        OperatorListBasis,
        OperatorMetadata,
    )


def _get_noise_eigenvalues_isotropic_fft[M: BasisMetadata](
    kernel: IsotropicKernelWithMetadata[M, np.dtype[np.complexfloating]],
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
        kernel.basis.outer_recast.outer_recast,
    )
    operators = operators.with_basis(as_tuple_basis(operators.basis))
    eigenvalues = _get_noise_eigenvalues_isotropic_fft(kernel)
    return OperatorList(
        TupleBasis((FundamentalBasis(eigenvalues), operators.basis.children[1])),
        operators.raw_data,
    )


def _get_noise_eigenvalues_isotropic_stacked_fft[M: BasisMetadata, E](
    kernel: IsotropicKernelWithMetadata[
        TupleMetadata[tuple[M, ...], E], Ctype[Never], np.dtype[np.complexfloating]
    ],
    *,
    fundamental_shape: tuple[int, ...] | None = None,
) -> EigenvalueMetadata:
    converted_outer = as_tuple_basis(kernel.basis.outer_recast.outer_recast)
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


def build_periodic_noise_operators[
    B: Basis[EigenvalueMetadata, Ctype[Never]],
    M: BasisMetadata,
    E,
](
    list_basis: B,
    inner_basis: TupleBasis[tuple[Basis[M], ...], E, Any],
) -> OperatorList[
    OperatorListBasis[
        EigenvalueMetadata, OperatorMetadata[TupleMetadata[tuple[M, ...], E]]
    ],
    np.dtype[np.complexfloating],
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
    operators = operators.with_basis(operators.basis.inner)

    return OperatorList(
        TupleBasis(
            (FundamentalBasis(list_basis.metadata()), operators.basis.children[1])
        ),
        operators.raw_data,
    ).with_list_basis(list_basis)


def get_periodic_noise_operators_isotropic_stacked_fft[M: BasisMetadata, E](
    kernel: IsotropicKernelWithMetadata[
        TupleMetadata[tuple[M, ...], E], Ctype[Never], np.dtype[np.complexfloating]
    ],
    *,
    fundamental_shape: tuple[int, ...] | None = None,
) -> OperatorList[
    OperatorListBasis[
        EigenvalueMetadata, OperatorMetadata[TupleMetadata[tuple[M, ...], E]]
    ],
    np.dtype[np.complexfloating],
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
    converted_outer = as_tuple_basis(kernel.basis.outer_recast.inner_recast)
    converted = kernel.with_isotropic_basis(converted_outer)
    fundamental_shape = converted_outer.shape

    eigenvalues = _get_noise_eigenvalues_isotropic_stacked_fft(
        converted, fundamental_shape=fundamental_shape
    )
    return build_periodic_noise_operators(
        FundamentalBasis(eigenvalues), converted_outer
    )
