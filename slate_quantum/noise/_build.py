from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from scipy.constants import Boltzmann, hbar  # type: ignore stubs
from slate.basis import CoordinateBasis, as_index_basis, as_tuple_basis
from slate.metadata import AxisDirections

from slate_quantum import operator
from slate_quantum.metadata import EigenvalueMetadata
from slate_quantum.noise._kernel import (
    DiagonalNoiseKernel,
    IsotropicNoiseKernel,
    NoiseKernel,
)
from slate_quantum.noise.diagonalize._eigenvalue import (
    get_periodic_noise_operators_diagonal_eigenvalue,
    get_periodic_noise_operators_eigenvalue,
)
from slate_quantum.operator import SuperOperatorMetadata, get_commutator_operator_list

if TYPE_CHECKING:
    from collections.abc import Iterable

    from slate.metadata import (
        BasisMetadata,
        StackedMetadata,
    )
    from slate.metadata.length import LengthMetadata, SpacedLengthMetadata

    from slate_quantum.operator import (
        Operator,
        OperatorList,
    )

    from ._kernel import AxisKernel


def build_isotropic_kernel_from_function[M: LengthMetadata](
    metadata: M,
    fn: Callable[
        [np.ndarray[Any, np.dtype[np.float64]]],
        np.ndarray[Any, np.dtype[np.complex128]],
    ],
) -> IsotropicNoiseKernel[M, np.complex128]:
    """
    Get an Isotropic Kernel with a correllation beta(x-x').

    Parameters
    ----------
    basis : StackedBasisWithVolumeLike
    fn : Callable[
        [np.ndarray[Any, np.dtype[np.float64]]],
        np.ndarray[Any, np.dtype[np.complex128]],
    ]
        beta(x-x'), the correllation as a function of displacement

    Returns
    -------
    IsotropicNoiseKernel[
        TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],
    ]
    """
    displacements = operator.build.x_displacement_operator(metadata)
    correlation = fn(displacements.raw_data.reshape(displacements.basis.shape)[0])

    return IsotropicNoiseKernel(displacements.basis[0], correlation)


def build_isotropic_kernel_from_function_stacked[
    M: SpacedLengthMetadata,
    E: AxisDirections,
](
    metadata: StackedMetadata[M, E],
    fn: Callable[
        [np.ndarray[Any, np.dtype[np.float64]]],
        np.ndarray[Any, np.dtype[np.complex128]],
    ],
) -> IsotropicNoiseKernel[StackedMetadata[M, E], np.complex128]:
    """
    Get an Isotropic Kernel with a correllation beta(x-x').

    Parameters
    ----------
    basis : StackedBasisWithVolumeLike
    fn : Callable[
        [np.ndarray[Any, np.dtype[np.float64]]],
        np.ndarray[Any, np.dtype[np.complex128]],
    ]
        beta(x-x'), the correllation as a function of displacement

    Returns
    -------
    IsotropicNoiseKernel[
        TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],
    ]
    """
    displacements = operator.build.total_x_displacement_operator(metadata)
    correlation = fn(displacements.raw_data.reshape(displacements.basis.shape)[0])

    return IsotropicNoiseKernel(displacements.basis[0], correlation)


def build_axis_kernel_from_function_stacked[M: SpacedLengthMetadata](
    metadata: StackedMetadata[M, Any],
    fn: Callable[
        [np.ndarray[Any, np.dtype[np.float64]]],
        np.ndarray[Any, np.dtype[np.complex128]],
    ],
) -> AxisKernel[M, np.complex128]:
    """
    Get an Isotropic Kernel with a correllation beta(x-x').

    Parameters
    ----------
    basis : StackedBasisWithVolumeLike
    fn : Callable[
        [np.ndarray[Any, np.dtype[np.float64]]],
        np.ndarray[Any, np.dtype[np.complex128]],
    ]
        beta(x-x'), the correllation as a function of displacement

    Returns
    -------
    IsotropicNoiseKernel[
        TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],
    ]
    """
    return tuple(
        build_isotropic_kernel_from_function(child, fn) for child in metadata.children
    )


def gaussian_correllation_fn(
    a: float, sigma: float
) -> Callable[
    [np.ndarray[Any, np.dtype[np.float64]]], np.ndarray[Any, np.dtype[np.complex128]]
]:
    r"""Get a correllation function for a gaussian noise kernel.

    A gaussian noise kernel is isotropic, and separable into individual
    axis kernels. The kernel is given by

    .. math::
        \beta(x, x') = a^2 \exp{\frac{(-(x-x')^2 }{(2  \lambda^2))}}
    """

    def fn(
        displacements: np.ndarray[Any, np.dtype[np.float64]],
    ) -> np.ndarray[Any, np.dtype[np.complex128]]:
        return a**2 * np.exp(-(displacements**2) / (2 * sigma**2)).astype(
            np.complex128,
        )

    return fn


def lorentzian_correllation_fn(
    a: float, lambda_: float
) -> Callable[
    [np.ndarray[Any, np.dtype[np.float64]]], np.ndarray[Any, np.dtype[np.complex128]]
]:
    r"""Get a correllation function for a lorentzian noise kernel.

    A lorentzian noise kernel is isotropic, and separable into individual
    axis kernels. The kernel is given by

    .. math::
        \beta(x, x') = a^2 \frac{\lambda^2}{(x-x')^2 + \lambda^2}
    """

    def fn(
        displacements: np.ndarray[Any, np.dtype[np.float64]],
    ) -> np.ndarray[Any, np.dtype[np.complex128]]:
        return a**2 * lambda_**2 / (displacements**2 + lambda_**2).astype(np.complex128)

    return fn


def caldeira_leggett_correllation_fn(
    a: float, lambda_: float
) -> Callable[
    [np.ndarray[Any, np.dtype[np.float64]]], np.ndarray[Any, np.dtype[np.complex128]]
]:
    r"""Get a correllation function for a lorentzian noise kernel.

    A lorentzian noise kernel is isotropic, and separable into individual
    axis kernels. The kernel is given by

    .. math::
        \beta(x, x') = a^2 - \frac{\lambda^2}{4} (x-x')^2
    """

    def fn(
        displacements: np.ndarray[Any, np.dtype[np.float64]],
    ) -> np.ndarray[Any, np.dtype[np.complex128]]:
        return (a**2 - (lambda_**2 / 4) * displacements**2).astype(np.complex128)

    return fn


def get_temperature_corrected_operators[M0: BasisMetadata, M1: BasisMetadata](
    hamiltonian: Operator[M1, np.complex128],
    operators: OperatorList[M0, M1, np.complex128],
    temperature: float,
    eta: float,
) -> OperatorList[M0, M1, np.complex128]:
    """Get the temperature corrected operators."""
    commutator = get_commutator_operator_list(hamiltonian, operators)
    correction = commutator * (-1 * np.sqrt(eta / (8 * Boltzmann * temperature)))
    operators *= np.sqrt(2 * eta * Boltzmann * temperature / hbar**2)
    return correction + operators


def truncate_noise_operator_list[M0: EigenvalueMetadata, M1: BasisMetadata](
    operators: OperatorList[M0, M1, np.complex128],
    truncation: Iterable[int],
) -> OperatorList[M0, M1, np.complex128]:
    """
    Get a truncated list of diagonal operators.

    Parameters
    ----------
    operators : DiagonalNoiseOperatorList[FundamentalBasis[BasisMetadata], _B0, _B1]
    truncation : Iterable[int]

    Returns
    -------
    DiagonalNoiseOperatorList[FundamentalBasis[BasisMetadata], _B0, _B1]
    """
    converted = operators.with_basis(as_tuple_basis(operators.basis))
    converted_list = converted.with_list_basis(as_index_basis(converted.basis[0]))
    eigenvalues = (
        converted_list.basis.metadata()
        .children[0]
        .values[converted_list.basis[0].points]
    )
    args = np.argsort(np.abs(eigenvalues))[::-1][np.array(list(truncation))]
    list_basis = CoordinateBasis(args, as_tuple_basis(operators.basis)[0])
    return operators.with_list_basis(list_basis)


def truncate_noise_kernel[M: SuperOperatorMetadata](
    kernel: NoiseKernel[M, np.complex128],
    truncation: Iterable[int],
) -> NoiseKernel[M, np.complex128]:
    """
    Given a noise kernel, retain only the first n noise operators.

    Parameters
    ----------
    kernel : NoiseKernel[_B0, _B1, _B0, _B1]
    n : int

    Returns
    -------
    NoiseKernel[_B0, _B1, _B0, _B1]
    """
    operators = get_periodic_noise_operators_eigenvalue(kernel)

    truncated = truncate_noise_operator_list(operators, truncation)
    return NoiseKernel.from_operators(truncated)


def truncate_diagonal_noise_kernel[
    M: BasisMetadata,
](
    kernel: DiagonalNoiseKernel[M, np.complex128],
    truncation: Iterable[int],
) -> DiagonalNoiseKernel[M, np.complex128]:
    """
    Given a noise kernel, retain only the first n noise operators.

    Parameters
    ----------
    kernel : NoiseKernel[_B0, _B1, _B0, _B1]
    n : int

    Returns
    -------
    NoiseKernel[_B0, _B1, _B0, _B1]
    """
    operators = get_periodic_noise_operators_diagonal_eigenvalue(kernel)

    truncated = truncate_noise_operator_list(operators, truncation)
    return DiagonalNoiseKernel.from_operators(truncated)
