from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.constants import Boltzmann, hbar  # type: ignore stubs
from slate.basis import DiagonalBasis, TupleBasis2D, as_tuple_basis
from slate.metadata import (
    AxisDirections,
    SpacedLengthMetadata,
    VolumeMetadata,
    shallow_shape_from_nested,
)
from slate.metadata.volume import fundamental_stacked_delta_x

from slate_quantum.noise.build import (
    isotropic_kernel_from_function_stacked,
    lorentzian_correlation_fn,
)
from slate_quantum.noise.diagonalize._taylor import (
    get_periodic_noise_operators_explicit_taylor_expansion,
)

if TYPE_CHECKING:
    from slate import StackedMetadata
    from slate.basis import Basis, FundamentalBasis

    from slate_quantum.metadata import EigenvalueMetadata
    from slate_quantum.noise._kernel import IsotropicNoiseKernel
    from slate_quantum.operator._operator import OperatorList


def get_effective_lorentzian_parameter(
    metadata: VolumeMetadata,
    eta: float,
    temperature: float,
    *,
    lambda_factor: float = 2 * np.sqrt(2),
) -> tuple[float, float]:
    """Generate a set of lorentzian parameters A, Lambda for a friction coefficient eta."""
    smallest_max_displacement = (
        np.min(np.linalg.norm(fundamental_stacked_delta_x(metadata), axis=1)) / 2
    )
    lambda_ = smallest_max_displacement / lambda_factor
    # mu = A / lambda
    mu = np.sqrt(2 * eta * Boltzmann * temperature / hbar**2)
    a = mu * lambda_
    return (a, lambda_)


def get_lorentzian_isotropic_noise_kernel[M: SpacedLengthMetadata, E: AxisDirections](
    metadata: StackedMetadata[M, E],
    a: float,
    lambda_: float,
) -> IsotropicNoiseKernel[StackedMetadata[M, E], np.complexfloating]:
    """Get an isotropic noise kernel for a lorentzian correlation.

    beta(x,x') = a**2 * lambda_**2 / ((x-x')**2 + lambda_**2)

    Parameters
    ----------
    basis : TupleBasisWithLengthLike[*B_0s]
    a : float
    lambda_ : float

    Returns
    -------
    IsotropicNoiseKernel[
    TupleBasisLike[*tuple[FundamentalPositionBasis, ...]],
    ]
    """
    return isotropic_kernel_from_function_stacked(
        metadata, lorentzian_correlation_fn(a, lambda_)
    )


def _get_explicit_taylor_coefficients_lorentzian(
    a: float,
    lambda_: float,
    *,
    n_terms: int = 1,
) -> np.ndarray[Any, np.dtype[np.floating]]:
    i = np.arange(0, n_terms + 1)
    return a**2 * ((-1 / (lambda_**2)) ** i)


def get_lorentzian_operators_explicit_taylor[M: VolumeMetadata, DT: np.generic](
    a: float,
    lambda_: float,
    basis: Basis[M, DT],
    *,
    n_terms: int | None = None,
) -> OperatorList[
    EigenvalueMetadata,
    M,
    np.complexfloating,
    TupleBasis2D[
        np.complexfloating,
        FundamentalBasis[EigenvalueMetadata],
        DiagonalBasis[
            np.complexfloating,
            Basis[M, np.generic],
            Basis[M, np.generic],
            None,
        ],
        None,
    ],
]:
    """Calculate the noise operators for an isotropic lorentzian noise kernel, using an explicit Taylor expansion.

    This function makes use of the analytical expression for the Taylor expansion of lorentzian
    noise lambda_/(x^2 + lambda_^2) about origin to find the 2n+1 lowest fourier coefficients.

    Parameters
    ----------
    lambda_: float, the width of the lorentzian noise kernel
    basis: TupleBasisWithLengthLike[FundamentalPositionBasis]
    n: int, by default 1

    Return in the order of [const term, first n sine terms, first n cos terms]
    and also their corresponding coefficients.
    """
    # currently only support 1D
    assert len(shallow_shape_from_nested(basis.fundamental_shape)) == 1
    basis_x = as_tuple_basis(basis)
    n_terms = (basis_x[0].size // 2) if n_terms is None else n_terms

    # expand gaussian and define array containing coefficients for each term in the polynomial
    # coefficients for the explicit Taylor expansion of the gaussian noise
    # Normalize lambda
    delta_x = np.linalg.norm(fundamental_stacked_delta_x(basis.metadata()), axis=1)
    normalized_lambda = 2 * np.pi * lambda_ / delta_x
    polynomial_coefficients = _get_explicit_taylor_coefficients_lorentzian(
        a, normalized_lambda.item(), n_terms=n_terms
    )

    return cast(
        "Any",
        get_periodic_noise_operators_explicit_taylor_expansion(
            basis_x, polynomial_coefficients, n_terms=n_terms
        ),
    )
