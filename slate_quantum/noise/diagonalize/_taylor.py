from __future__ import annotations

from typing import TYPE_CHECKING, Any, Never, cast

import numpy as np
from scipy.special import factorial  # type: ignore library type
from slate_core.basis import (
    AsUpcast,
    Basis,
    DiagonalBasis,
    FundamentalBasis,
    TruncatedBasis,
    Truncation,
    TupleBasis,
    from_metadata,
)
from slate_core.metadata import (
    BasisMetadata,
)
from slate_core.util import pad_ft_points

from slate_quantum import operator
from slate_quantum.metadata import EigenvalueMetadata
from slate_quantum.noise._kernel import (
    IsotropicKernelWithMetadata,
    get_diagonal_noise_operators_from_axis,
)
from slate_quantum.noise.build._build import axis_kernel_from_isotropic
from slate_quantum.operator import (
    OperatorList,
)

if TYPE_CHECKING:
    from slate_core import Ctype, TupleMetadata

    from slate_quantum.operator._operator import (
        OperatorListBasis,
        OperatorListMetadata,
        OperatorMetadata,
    )


def _get_cos_coefficients_for_taylor_series(
    polynomial_coefficients: np.ndarray[Any, np.dtype[np.floating]],
    *,
    n_terms: int | None = None,
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
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
    coefficients_prefactor = ((-1) ** m) / cast("float", factorial(2 * m))
    coefficients_matrix = coefficients_prefactor * (i ** (2 * m))
    cos_series_coefficients = np.linalg.solve(
        coefficients_matrix, polynomial_coefficients
    )
    out = np.zeros(n_terms, np.floating)
    out[:n_nonzero_terms] = cos_series_coefficients.T
    return out


def _get_periodic_coefficients_for_taylor_series(
    polynomial_coefficients: np.ndarray[Any, np.dtype[np.floating]],
    *,
    n_terms: int | None = None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
    cos_series_coefficients = _get_cos_coefficients_for_taylor_series(
        polynomial_coefficients, n_terms=n_terms
    )
    sin_series_coefficients = cos_series_coefficients[:0:-1]
    return np.concatenate([cos_series_coefficients, sin_series_coefficients])


def get_periodic_noise_operators_explicit_taylor_expansion[
    M: BasisMetadata,
](
    basis: Basis[M, Any],
    polynomial_coefficients: np.ndarray[tuple[int], np.dtype[np.floating]],
    *,
    n_terms: int | None = None,
) -> OperatorList[
    OperatorListBasis[EigenvalueMetadata, OperatorMetadata[M]],
    np.dtype[np.complexfloating],
]:
    """Note polynomial_coefficients should be properly normalized."""
    n_terms = (basis.size // 2) if n_terms is None else n_terms

    operators = operator.build.all_axis_periodic_operators(basis)
    list_basis = TupleBasis(
        (
            TruncatedBasis(
                Truncation(n_terms, 0, 0),
                from_metadata(operators.basis.metadata().children[0]),
            ).upcast(),
            operators.basis.inner.children[1],
        )
    )
    converted = operators.with_basis(list_basis.upcast()).assert_ok()

    # coefficients for the Taylor expansion of the trig terms
    coefficients = _get_periodic_coefficients_for_taylor_series(
        polynomial_coefficients=polynomial_coefficients,
    )
    coefficients *= basis.size

    eigenvalues = EigenvalueMetadata(coefficients.astype(np.complex128))

    return OperatorList.build(
        TupleBasis(
            (FundamentalBasis(eigenvalues), converted.basis.inner.children[1])
        ).upcast(),
        converted.raw_data,
    ).assert_ok()


def _get_linear_operators_for_noise[M: BasisMetadata](
    metadata: M, *, n_terms: int | None = None
) -> OperatorList[
    AsUpcast[
        TupleBasis[
            tuple[
                FundamentalBasis,
                AsUpcast[
                    DiagonalBasis[TupleBasis[tuple[Basis[M], Basis[M]], None]],
                    OperatorMetadata[M],
                ],
            ],
            None,
        ],
        OperatorListMetadata,
    ],
    np.dtype[np.complexfloating],
]:
    size = np.prod(metadata.fundamental_shape).item()
    n_terms = size if n_terms is None else n_terms

    nx_displacements = operator.build.nx_displacement_operator(metadata)

    displacements = (
        2
        * np.pi
        * nx_displacements.raw_data.reshape(nx_displacements.basis.inner.shape)[
            size // 2
        ]
        / size
    )
    data = np.array([displacements**n for n in range(n_terms)], dtype=np.complex128)

    return OperatorList.build(
        TupleBasis(
            (
                FundamentalBasis.from_size(n_terms),
                DiagonalBasis(nx_displacements.basis.inner).upcast(),
            )
        ).upcast(),
        data,
    ).assert_ok()


def get_linear_noise_operators_explicit_taylor_expansion[M: BasisMetadata](
    metadata: M,
    polynomial_coefficients: np.ndarray[tuple[int], np.dtype[np.floating]],
    *,
    n_terms: int | None = None,
) -> OperatorList[
    OperatorListBasis[EigenvalueMetadata, OperatorMetadata[M]],
    np.dtype[np.complexfloating],
]:
    """Note polynomial_coefficients should be properly normalized.

    This is done such that 'x' changes between 0 and 2pi
    """
    operators = _get_linear_operators_for_noise(metadata, n_terms=n_terms)
    eigenvalues = EigenvalueMetadata(polynomial_coefficients.astype(np.complex128))

    return OperatorList.build(
        TupleBasis(
            (FundamentalBasis(eigenvalues), operators.basis.inner.children[1])
        ).upcast(),
        operators.raw_data,
    ).assert_ok()


def get_periodic_noise_operators_real_isotropic_taylor_expansion[M: BasisMetadata](
    kernel: IsotropicKernelWithMetadata[M, Ctype[Never], np.dtype[np.complexfloating]],
    *,
    n: int | None = None,
) -> OperatorList[
    OperatorListBasis[EigenvalueMetadata, OperatorMetadata[M]],
    np.dtype[np.complexfloating],
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
        "np.polynomial.Polynomial",
        np.polynomial.Chebyshev.fit(  # cSpell: ignore Chebyshev # type: ignore unknown
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
    operators = None
    msg = "Need to implement for e^ikx operators."
    raise NotImplementedError(msg)
    return OperatorList(
        TupleBasis((FundamentalBasis(eigenvalues), operators.basis[1])),
        operators.raw_data,
    )


def get_periodic_noise_operators_real_isotropic_stacked_taylor_expansion[
    M: BasisMetadata,
    E,
](
    kernel: IsotropicKernelWithMetadata[
        TupleMetadata[tuple[M, ...], E],
        Ctype[Never],
        np.dtype[np.complexfloating],
    ],
    *,
    shape: tuple[int | None, ...] | None = None,
) -> OperatorList[
    OperatorListBasis[
        EigenvalueMetadata, OperatorMetadata[TupleMetadata[tuple[M, ...], E]]
    ],
    np.dtype[np.complexfloating],
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
    axis_kernels = axis_kernel_from_isotropic(kernel)

    operators_list = tuple(
        get_periodic_noise_operators_real_isotropic_taylor_expansion(
            kernel,
            n=None if shape is None else shape[i],
        )
        for (i, kernel) in enumerate(axis_kernels)
    )
    extra = kernel.basis.inner.outer_recast.outer_recast.metadata().extra
    return get_diagonal_noise_operators_from_axis(operators_list, extra)  # type: ignore bad inference
