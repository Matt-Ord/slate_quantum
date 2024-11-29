from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.constants import Boltzmann, hbar  # type:ignore bad stub file
from scipy.special import factorial  # type:ignore bad stub file
from slate.basis import FundamentalBasis
from slate.metadata import (
    AxisDirections,
    SpacedVolumeMetadata,
    StackedMetadata,
)
from slate.metadata.length import SpacedLengthMetadata
from slate.metadata.volume import fundamental_stacked_delta_x

from slate_quantum.noise._build import (
    build_axis_kernel_from_function_stacked,
    build_isotropic_kernel_from_function_stacked,
    gaussian_correllation_fn,
    get_temperature_corrected_operators,
    truncate_noise_operator_list,
)
from slate_quantum.noise._kernel import get_diagonal_noise_operators_from_axis
from slate_quantum.noise.diagonalize._fft import (
    get_periodic_noise_operators_real_isotropic_stacked_fft,
)
from slate_quantum.noise.diagonalize._taylor import (
    get_linear_noise_operators_explicit_taylor_expansion,
    get_periodic_noise_operators_explicit_taylor_expansion,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from slate.basis import Basis, DiagonalBasis, TupleBasis2D

    from slate_quantum.metadata import EigenvalueMetadata
    from slate_quantum.noise._kernel import IsotropicNoiseKernel
    from slate_quantum.operator._operator import Operator, OperatorList

    from ._kernel import AxisKernel, DiagonalNoiseKernel


def get_gaussian_isotropic_noise_kernel[
    M: SpacedLengthMetadata,
    E: AxisDirections,
](
    metadata: StackedMetadata[M, E],
    a: float,
    lambda_: float,
) -> IsotropicNoiseKernel[StackedMetadata[M, E], np.complex128]:
    """Get the noise kernel for a gaussian correllated surface."""
    return build_isotropic_kernel_from_function_stacked(
        metadata, gaussian_correllation_fn(a, lambda_)
    )


def get_gaussian_axis_noise_kernel[M: SpacedLengthMetadata](
    metadata: StackedMetadata[M, Any],
    a: float,
    lambda_: float,
) -> AxisKernel[M, np.complex128]:
    """Get the noise kernel for a gaussian correllated surface."""
    return build_axis_kernel_from_function_stacked(
        metadata, gaussian_correllation_fn(a, lambda_)
    )


def get_gaussian_noise_kernel[
    M: SpacedLengthMetadata,
    E: AxisDirections,
](
    metadata: StackedMetadata[M, E],
    a: float,
    lambda_: float,
) -> DiagonalNoiseKernel[StackedMetadata[M, E], np.complex128]:
    """Get the noise kernel for a gaussian correllated surface."""
    return get_gaussian_isotropic_noise_kernel(metadata, a, lambda_).unwrap()


def get_effective_gaussian_parameters(
    metadata: SpacedVolumeMetadata,
    eta: float,
    temperature: float,
    *,
    lambda_factor: float = 2 * np.sqrt(2),
) -> tuple[float, float]:
    """
    Generate a set of Gaussian parameters A, Lambda for a friction coefficient eta.

    This is done to match the quadratic coefficient (A^2/(2 lambda^2))
    beta(x,x') = A^2(1-(x-x')^2/(lambda^2))

    to the caldeira leggett noise

    beta(x,x') = 2 * eta * Boltzmann * temperature / hbar**2
    """
    smallest_max_displacement = np.min(
        np.linalg.norm(fundamental_stacked_delta_x(metadata), axis=1)
    )
    lambda_ = smallest_max_displacement / lambda_factor
    # mu = A / lambda
    mu = np.sqrt(2 * eta * Boltzmann * temperature / hbar**2)
    a = mu * lambda_
    return (a, lambda_)


def get_effective_gaussian_noise_kernel[
    M: SpacedLengthMetadata,
    E: AxisDirections,
](
    metadata: StackedMetadata[M, E],
    eta: float,
    temperature: float,
    *,
    lambda_factor: float = 2 * np.sqrt(2),
) -> DiagonalNoiseKernel[StackedMetadata[M, E], np.complex128]:
    """
    Get the noise kernel for a gaussian correllated surface, given the Caldeira leggett parameters.

    This chooses the largest possible wavelength, such that the smallest correllation between
    any two points is a**2 * np.exp(- lambda_factor ** 2 / 2), where a**2 is the max correllation

    Parameters
    ----------
    basis : TupleBasisLike[BasisWithLengthLike[Any, Any, Literal[1]]]
    eta : float
    temperature : float
    lambda_factor : float, optional
        lambda_factor, by default 2*np.sqrt(2)

    Returns
    -------
    SingleBasisDiagonalNoiseKernel[ TupleBasisLike[FundamentalPositionBasis] ]
    """
    a, lambda_ = get_effective_gaussian_parameters(
        metadata, eta, temperature, lambda_factor=lambda_factor
    )
    return get_gaussian_noise_kernel(metadata, a, lambda_)


def get_effective_gaussian_isotropic_noise_kernel[
    M: SpacedLengthMetadata,
    E: AxisDirections,
](
    metadata: StackedMetadata[M, E],
    eta: float,
    temperature: float,
    *,
    lambda_factor: float = 2 * np.sqrt(2),
) -> IsotropicNoiseKernel[StackedMetadata[M, E], np.complex128]:
    """
    Get the noise kernel for a gaussian correllated surface, given the Caldeira leggett parameters.

    This chooses the largest possible wavelength, such that the smallest correllation between
    any two points is a**2 * np.exp(- lambda_factor ** 2 / 2), where a**2 is the max correllation

    Parameters
    ----------
    basis : TupleBasisLike[BasisWithLengthLike[Any, Any, Literal[1]]]
    eta : float
    temperature : float
    lambda_factor : float, optional
        lambda_factor, by default 2*np.sqrt(2)

    Returns
    -------
    SingleBasisDiagonalNoiseKernel[ TupleBasisLike[FundamentalPositionBasis] ]
    """
    a, lambda_ = get_effective_gaussian_parameters(
        metadata, eta, temperature, lambda_factor=lambda_factor
    )
    return get_gaussian_isotropic_noise_kernel(metadata, a, lambda_)


def get_gaussian_noise_operators_periodic[
    M: SpacedLengthMetadata,
    E: AxisDirections,
](
    metadata: StackedMetadata[M, E],
    a: float,
    lambda_: float,
    *,
    truncation: Iterable[int] | None = None,
) -> OperatorList[
    EigenvalueMetadata,
    StackedMetadata[M, E],
    np.complex128,
]:
    """Get the noise operators for a gausssian kernel in the given basis.

    Parameters
    ----------
    hamiltonian : SingleBasisOperator[_BL0]
    mass : float
    temperature : float
    gamma : float

    Returns
    -------
    SingleBasisNoiseOperatorList[
        FundamentalBasis[BasisMetadata],
        FundamentalPositionBasis,
    ]

    """
    kernel = get_gaussian_isotropic_noise_kernel(metadata, a, lambda_)

    operators = get_periodic_noise_operators_real_isotropic_stacked_fft(kernel)
    truncation = range(operators.basis[0].size) if truncation is None else truncation
    return truncate_noise_operator_list(operators, truncation=truncation)


def get_effective_gaussian_noise_operators_periodic[
    M: SpacedLengthMetadata,
    E: AxisDirections,
](
    metadata: StackedMetadata[M, E],
    eta: float,
    temperature: float,
    *,
    truncation: Iterable[int] | None = None,
) -> OperatorList[
    EigenvalueMetadata,
    StackedMetadata[M, E],
    np.complex128,
]:
    """Get the noise operators for a gausssian kernel in the given basis.

    Parameters
    ----------
    hamiltonian : SingleBasisOperator[_BL0]
    mass : float
    temperature : float
    gamma : float

    Returns
    -------
    SingleBasisNoiseOperatorList[
        FundamentalBasis[BasisMetadata],
        FundamentalPositionBasis,
    ]

    """
    a, lambda_ = get_effective_gaussian_parameters(metadata, eta, temperature)
    return get_gaussian_noise_operators_periodic(
        metadata, a, lambda_, truncation=truncation
    )


def get_temperature_corrected_gaussian_noise_operators[
    M: SpacedLengthMetadata,
    E: AxisDirections,
](
    hamiltonian: Operator[StackedMetadata[M, E], np.complex128],
    a: float,
    lambda_: float,
    temperature: float,
    *,
    truncation: Iterable[int] | None = None,
) -> OperatorList[
    EigenvalueMetadata,
    StackedMetadata[M, E],
    np.complex128,
]:
    """Get the noise operators for a gausssian kernel in the given basis.

    Parameters
    ----------
    hamiltonian : SingleBasisOperator[_BL0]
    mass : float
    temperature : float
    gamma : float

    Returns
    -------
    SingleBasisNoiseOperatorList[
        FundamentalBasis[BasisMetadata],
        FundamentalPositionBasis,
    ]

    """
    operators = get_gaussian_noise_operators_periodic(
        hamiltonian.basis.metadata()[0], a, lambda_, truncation=truncation
    )
    corrected = get_temperature_corrected_operators(hamiltonian, operators, temperature)
    return corrected.with_operator_basis(hamiltonian.basis)


def get_temperature_corrected_effective_gaussian_noise_operators[
    M: SpacedVolumeMetadata,
](
    hamiltonian: Operator[M, np.complex128],
    eta: float,
    temperature: float,
    *,
    truncation: Iterable[int] | None = None,
) -> OperatorList[
    EigenvalueMetadata,
    SpacedVolumeMetadata,
    np.complex128,
]:
    """Get the noise operators for a gausssian kernel in the given basis.

    Parameters
    ----------
    hamiltonian : SingleBasisOperator[_BL0]
    mass : float
    temperature : float
    gamma : float

    Returns
    -------
    SingleBasisNoiseOperatorList[
        FundamentalBasis[BasisMetadata],
        FundamentalPositionBasis,
    ]

    """
    a, lambda_ = get_effective_gaussian_parameters(
        hamiltonian.basis.metadata()[0], eta, temperature
    )

    return get_temperature_corrected_gaussian_noise_operators(
        hamiltonian, a, lambda_, temperature, truncation=truncation
    )


def _get_explicit_taylor_coefficients_gaussian(
    a: float,
    lambda_: float,
    *,
    n_terms: int = 1,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    i = np.arange(0, n_terms)
    return (a**2 / factorial(i)) * ((-1 / (2 * lambda_**2)) ** i)


def get_periodic_gaussian_operators_explicit_taylor[M: SpacedLengthMetadata](
    metadata: M,
    a: float,
    lambda_: float,
    *,
    n_terms: int | None = None,
) -> OperatorList[
    EigenvalueMetadata,
    M,
    np.complex128,
]:
    """Calculate the noise operators for an isotropic gaussian noise kernel, using an explicit Taylor expansion.

    This function makes use of the analytical expression for the Taylor expansion of gaussian
    noise (a^2)*e^(-x^2/2*lambda_^2) about origin to find the 2n+1 lowest fourier coefficients.

    Return in the order of [const term, first n cos terms, first n sin terms]
    and also their corresponding coefficients.
    """
    n_terms = (
        (np.prod(metadata.fundamental_shape).item() // 2)
        if n_terms is None
        else n_terms
    )

    # expand gaussian and define array containing coefficients for each term in the polynomial
    # coefficients for the explicit Taylor expansion of the gaussian noise
    # Normalize lambda
    delta_x = np.linalg.norm(metadata.delta)
    normalized_lambda = 2 * np.pi * lambda_ / delta_x
    polynomial_coefficients = _get_explicit_taylor_coefficients_gaussian(
        a, normalized_lambda.item(), n_terms=n_terms
    )

    basis_x = FundamentalBasis(metadata)
    return get_periodic_noise_operators_explicit_taylor_expansion(
        basis_x, polynomial_coefficients, n_terms=n_terms
    )


def get_linear_gaussian_noise_operators_explicit_taylor[M: SpacedLengthMetadata](
    metadata: M,
    a: float,
    lambda_: float,
    *,
    n_terms: int | None = None,
) -> OperatorList[
    EigenvalueMetadata,
    M,
    np.complex128,
    TupleBasis2D[
        np.complex128,
        FundamentalBasis[EigenvalueMetadata],
        DiagonalBasis[np.complex128, Basis[M, Any], Basis[M, Any], None],
        None,
    ],
]:
    """Get the noise operators for a gausssian kernel in the given basis.

    Parameters
    ----------
    hamiltonian : SingleBasisOperator[_BL0]
    mass : float
    temperature : float
    gamma : float

    Returns
    -------
    SingleBasisNoiseOperatorList[
        FundamentalBasis[BasisMetadata],
        FundamentalPositionBasis,
    ]

    """
    n_terms = (
        (np.prod(metadata.fundamental_shape).item()) if n_terms is None else n_terms
    )

    # expand gaussian and define array containing coefficients for each term in the polynomial
    # coefficients for the explicit Taylor expansion of the gaussian noise
    # Normalize lambda
    delta_x = np.linalg.norm(metadata.delta)
    normalized_lambda = 2 * np.pi * lambda_ / delta_x
    polynomial_coefficients = _get_explicit_taylor_coefficients_gaussian(
        a, normalized_lambda.item(), n_terms=n_terms
    )
    return get_linear_noise_operators_explicit_taylor_expansion(
        metadata, polynomial_coefficients, n_terms=n_terms
    )


def get_periodic_gaussian_operators_explicit_taylor_stacked[M: SpacedLengthMetadata, E](
    metadata: StackedMetadata[M, E],
    a: float,
    lambda_: float,
    *,
    shape: tuple[int, ...] | None = None,
) -> OperatorList[EigenvalueMetadata, StackedMetadata[M, E], np.complex128]:
    """Calculate the noise operators for an isotropic gaussian noise kernel, using an explicit Taylor expansion.

    This function makes use of the analytical expression for the Taylor expansion of gaussian
    noise (a^2)*e^(-x^2/2*lambda_^2) about origin to find the 2n+1 lowest fourier coefficients.

    Return in the order of [const term, first n cos terms, first n sin terms]
    and also their corresponding coefficients.
    """
    axis_operators = tuple(
        get_periodic_gaussian_operators_explicit_taylor(
            child,
            a,
            lambda_,
            n_terms=None if shape is None else shape[i],
        )
        for i, child in enumerate(metadata.children)
    )

    return get_diagonal_noise_operators_from_axis(axis_operators, metadata.extra)


def get_linear_gaussian_operators_explicit_taylor_stacked[M: SpacedLengthMetadata, E](
    metadata: StackedMetadata[M, E],
    a: float,
    lambda_: float,
    *,
    shape: tuple[int, ...] | None = None,
) -> OperatorList[EigenvalueMetadata, StackedMetadata[M, E], np.complex128]:
    """Calculate the noise operators for an isotropic gaussian noise kernel, using an explicit Taylor expansion.

    This function makes use of the analytical expression for the Taylor expansion of gaussian
    noise (a^2)*e^(-x^2/2*lambda_^2) about origin to find the 2n+1 lowest fourier coefficients.

    Return in the order of [const term, first n cos terms, first n sin terms]
    and also their corresponding coefficients.
    """
    axis_operators = tuple(
        get_linear_gaussian_noise_operators_explicit_taylor(
            child,
            a,
            lambda_,
            n_terms=None if shape is None else shape[i],
        )
        for i, child in enumerate(metadata.children)
    )

    return get_diagonal_noise_operators_from_axis(axis_operators, metadata.extra)
