from __future__ import annotations

from typing import Any, cast

import numpy as np
from scipy.special import factorial  # type: ignore library type
from slate.basis import (
    Basis,
    DiagonalBasis,
    FundamentalBasis,
    TupleBasis2D,
    tuple_basis,
)
from slate.metadata import BasisMetadata, Metadata2D, SimpleMetadata, StackedMetadata
from slate.util import pad_ft_points

from slate_quantum.model._label import EigenvalueMetadata
from slate_quantum.model.operator import (
    OperatorList,
)
from slate_quantum.model.operator.build._displacement import (
    build_nx_displacement_operator,
)
from slate_quantum.noise.diagonalization._fft import (
    get_periodic_operators_for_real_isotropic_noise,
)
from slate_quantum.noise.kernel import (
    IsotropicNoiseKernel,
    as_axis_kernel_from_isotropic,
    get_diagonal_noise_operators_from_axis,
)


def _get_cos_coefficients_for_taylor_series(
    polynomial_coefficients: np.ndarray[Any, np.dtype[np.float64]],
    *,
    n_terms: int | None = None,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    n_terms = polynomial_coefficients.size if n_terms is None else n_terms

    atol = 1e-8 * np.max(polynomial_coefficients).item()
    is_nonzero = np.isclose(polynomial_coefficients, 0, atol=atol)
    first_nonzero = np.argmax(is_nonzero).item()
    if first_nonzero == 0 and is_nonzero.item(0) is False:
        first_nonzero = is_nonzero.size
    n_nonzero_terms = min(first_nonzero, n_terms)
    polynomial_coefficients = polynomial_coefficients[:n_nonzero_terms]

    i = np.arange(0, n_nonzero_terms).reshape(1, -1)
    m = np.arange(0, n_nonzero_terms).reshape(-1, 1)
    coefficients_prefactor = ((-1) ** m) / (factorial(2 * m))
    coefficients_matrix = coefficients_prefactor * (i ** (2 * m))
    cos_series_coefficients = np.linalg.solve(
        coefficients_matrix, polynomial_coefficients
    )
    out = np.zeros(n_terms, np.float64)
    out[:n_nonzero_terms] = cos_series_coefficients.T
    return out


def _get_periodic_coefficients_for_taylor_series(
    polynomial_coefficients: np.ndarray[Any, np.dtype[np.float64]],
    *,
    n_terms: int | None = None,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    cos_series_coefficients = _get_cos_coefficients_for_taylor_series(
        polynomial_coefficients, n_terms=n_terms
    )
    sin_series_coefficients = cos_series_coefficients[:0:-1]
    return np.concatenate([cos_series_coefficients, sin_series_coefficients])


def get_periodic_noise_operators_explicit_taylor_expansion[
    M: BasisMetadata,
    B: Basis[BasisMetadata, Any] = Basis[M, Any],
](
    basis: B,
    polynomial_coefficients: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    n_terms: int | None = None,
) -> OperatorList[
    Metadata2D[EigenvalueMetadata, Metadata2D[M, M, None], None],
    np.complex128,
    TupleBasis2D[
        np.complex128,
        FundamentalBasis[EigenvalueMetadata],
        DiagonalBasis[np.complex128, B, B, None],
        None,
    ],
]:
    """Note polynomial_coefficients should be properly normalized."""
    n_terms = (basis.size // 2) if n_terms is None else n_terms

    operators = get_periodic_operators_for_real_isotropic_noise(basis, n=n_terms)

    # coefficients for the Taylor expansion of the trig terms
    coefficients = _get_periodic_coefficients_for_taylor_series(
        polynomial_coefficients=polynomial_coefficients,
    )
    coefficients *= basis.size

    eigenvalues = EigenvalueMetadata(coefficients.astype(np.complex128))

    return OperatorList(
        tuple_basis((FundamentalBasis(eigenvalues), operators.basis.children[1])),
        operators.raw_data,
    )


def _get_linear_operators_for_noise[M: BasisMetadata](
    metadata: M, *, n_terms: int | None = None
) -> OperatorList[
    Metadata2D[BasisMetadata, Metadata2D[M, M, None], None],
    np.complex128,
    TupleBasis2D[
        np.complex128,
        FundamentalBasis[SimpleMetadata],
        DiagonalBasis[np.complex128, FundamentalBasis[M], FundamentalBasis[M], None],
        None,
    ],
]:
    size = np.prod(metadata.fundamental_shape).item()
    n_terms = size if n_terms is None else n_terms

    nx_displacements = build_nx_displacement_operator(metadata)

    displacements = (
        2
        * np.pi
        * nx_displacements.raw_data.reshape(nx_displacements.basis.shape)[size // 2]
        / size
    )
    data = np.array([displacements**n for n in range(n_terms)], dtype=np.complex128)

    return OperatorList(
        tuple_basis(
            (
                FundamentalBasis.from_shape((n_terms,)),
                DiagonalBasis(nx_displacements.basis),
            ),
            None,
        ),
        data,
    )


def get_linear_noise_operators_explicit_taylor_expansion[M: BasisMetadata](
    metadata: M,
    polynomial_coefficients: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    n_terms: int | None = None,
) -> OperatorList[
    Metadata2D[EigenvalueMetadata, Metadata2D[M, M, None], None],
    np.complex128,
    TupleBasis2D[
        np.complex128,
        FundamentalBasis[EigenvalueMetadata],
        DiagonalBasis[np.complex128, FundamentalBasis[M], FundamentalBasis[M], None],
        None,
    ],
]:
    """Note polynomial_coefficients should be properly normalized.

    This is done such that 'x' changes between 0 and 2pi
    """
    operators = _get_linear_operators_for_noise(metadata, n_terms=n_terms)
    eigenvalues = EigenvalueMetadata(polynomial_coefficients.astype(np.complex128))

    return OperatorList(
        tuple_basis((FundamentalBasis(eigenvalues), operators.basis.children[1])),
        operators.raw_data,
    )


def get_periodic_noise_operators_real_isotropic_taylor_expansion[M: BasisMetadata](
    kernel: IsotropicNoiseKernel[M, np.complex128],
    *,
    n: int | None = None,
) -> OperatorList[
    Metadata2D[
        EigenvalueMetadata, Metadata2D[BasisMetadata, BasisMetadata, None], None
    ],
    np.complex128,
    TupleBasis2D[
        np.complex128,
        FundamentalBasis[EigenvalueMetadata],
        DiagonalBasis[np.complex128, Basis[M, Any], Basis[M, Any], None],
        None,
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
    n_states: int = kernel.basis.fundamental_size
    n = (n_states + 1) // 2 if n is None else n

    # weight is chosen such that the 2n+1 points around the origin are selected for fitting
    weight = pad_ft_points(np.ones(2 * n + 1), (n_states,), (0,))
    points = np.cos(np.arange(n_states) * (2 * np.pi / n_states))

    # use T_n(cos(x)) = cos(nx) to find the coefficients
    noise_polynomial = cast(
        np.polynomial.Polynomial,
        np.polynomial.Chebyshev.fit(  # type: ignore unknown
            x=points,
            y=kernel.raw_data,
            deg=n,
            w=weight,
            domain=(-1, 1),
        ),
    )

    operator_coefficients = np.concatenate(
        [noise_polynomial.coef, noise_polynomial.coef[:0:-1]]
    ).astype(np.complex128)
    operator_coefficients *= n_states
    eigenvalues = EigenvalueMetadata(operator_coefficients)
    operators = get_periodic_operators_for_real_isotropic_noise(
        kernel.basis.inner_recast, n=n + 1
    )

    return OperatorList(
        tuple_basis((FundamentalBasis(eigenvalues), operators.basis.children[1])),
        operators.raw_data,
    )


def get_periodic_noise_operators_real_isotropic_stacked_taylor_expansion[
    M: BasisMetadata
](
    kernel: IsotropicNoiseKernel[
        StackedMetadata[M, Any],
        np.complex128,
    ],
    *,
    shape: tuple[int | None, ...] | None = None,
) -> OperatorList[
    Metadata2D[
        EigenvalueMetadata,
        Metadata2D[
            StackedMetadata[BasisMetadata, Any],
            StackedMetadata[BasisMetadata, Any],
            None,
        ],
        None,
    ],
    np.complex128,
    Basis[
        Metadata2D[
            EigenvalueMetadata,
            Metadata2D[
                StackedMetadata[BasisMetadata, Any],
                StackedMetadata[BasisMetadata, Any],
                None,
            ],
            Any,
        ],
        np.complex128,
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

    operators_list = tuple(
        get_periodic_noise_operators_real_isotropic_taylor_expansion(
            kernel,
            n=None if shape is None else shape[i],
        )
        for (i, kernel) in enumerate(axis_kernels)
    )
    return get_diagonal_noise_operators_from_axis(
        operators_list, kernel.basis.outer_recast.metadata().extra
    )
