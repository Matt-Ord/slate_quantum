from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.constants import Boltzmann, hbar  # type: ignore stubs
from slate_core import basis, linalg
from slate_core.basis import CoordinateBasis
from slate_core.metadata import AxisDirections

from slate_quantum import operator
from slate_quantum.metadata import EigenvalueMetadata
from slate_quantum.noise._kernel import (
    DiagonalNoiseKernel,
    IsotropicNoiseKernel,
    NoiseKernel,
)
from slate_quantum.noise.diagonalize import (
    get_periodic_noise_operators_diagonal_eigenvalue,
    get_periodic_noise_operators_eigenvalue,
)
from slate_quantum.operator import (
    Operator,
    SuperOperatorMetadata,
    get_commutator_operator_list,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from slate_core.metadata import (
        BasisMetadata,
    )
    from slate_core.metadata.length import LengthMetadata, SpacedLengthMetadata

    from slate_quantum._util.legacy import StackedMetadata
    from slate_quantum.noise._kernel import AxisKernel
    from slate_quantum.operator import (
        OperatorList,
    )


def isotropic_kernel_from_function[M: LengthMetadata](
    metadata: M,
    fn: Callable[
        [np.ndarray[Any, np.dtype[np.float64]]],
        np.ndarray[Any, np.dtype[np.complexfloating]],
    ],
) -> IsotropicNoiseKernel[M, np.complexfloating]:
    """
    Get an Isotropic Kernel with a correlation beta(x-x').

    Parameters
    ----------
    basis : StackedBasisWithVolumeLike
    fn : Callable[
        [np.ndarray[Any, np.dtype[np.float64]]],
        np.ndarray[Any, np.dtype[np.complexfloating]],
    ]
        beta(x-x'), the correlation as a function of displacement

    Returns
    -------
    IsotropicNoiseKernel[
        TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],
    ]
    """
    displacements = operator.build.x_displacement_operator(metadata)
    correlation = fn(displacements.raw_data.reshape(displacements.basis.shape)[0])

    return IsotropicNoiseKernel(displacements.basis[0], correlation)


def isotropic_kernel_from_function_stacked[
    M: SpacedLengthMetadata,
    E: AxisDirections,
](
    metadata: StackedMetadata[M, E],
    fn: Callable[
        [np.ndarray[Any, np.dtype[np.float64]]],
        np.ndarray[Any, np.dtype[np.complexfloating]],
    ],
) -> IsotropicNoiseKernel[StackedMetadata[M, E], np.complexfloating]:
    """
    Get an Isotropic Kernel with a correlation beta(x-x').

    Parameters
    ----------
    basis : StackedBasisWithVolumeLike
    fn : Callable[
        [np.ndarray[Any, np.dtype[np.float64]]],
        np.ndarray[Any, np.dtype[np.complexfloating]],
    ]
        beta(x-x'), the correlation as a function of displacement

    Returns
    -------
    IsotropicNoiseKernel[
        TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],
    ]
    """
    displacements = operator.build.total_x_displacement_operator(metadata)
    correlation = fn(displacements.raw_data.reshape(displacements.basis.shape)[0])

    return IsotropicNoiseKernel(displacements.basis[0], correlation)


def axis_kernel_from_function_stacked[M: SpacedLengthMetadata](
    metadata: StackedMetadata[M, Any],
    fn: Callable[
        [np.ndarray[Any, np.dtype[np.float64]]],
        np.ndarray[Any, np.dtype[np.complexfloating]],
    ],
) -> AxisKernel[M, np.complexfloating]:
    """
    Get an Isotropic Kernel with a correlation beta(x-x').

    Parameters
    ----------
    basis : StackedBasisWithVolumeLike
    fn : Callable[
        [np.ndarray[Any, np.dtype[np.float64]]],
        np.ndarray[Any, np.dtype[np.complexfloating]],
    ]
        beta(x-x'), the correlation as a function of displacement

    Returns
    -------
    IsotropicNoiseKernel[
        TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],
    ]
    """
    return tuple(
        isotropic_kernel_from_function(child, fn) for child in metadata.children
    )


def gaussian_correlation_fn(
    a: float, sigma: float
) -> Callable[
    [np.ndarray[Any, np.dtype[np.float64]]],
    np.ndarray[Any, np.dtype[np.complexfloating]],
]:
    r"""Get a correlation function for a gaussian noise kernel.

    A gaussian noise kernel is isotropic, and separable into individual
    axis kernels. The kernel is given by

    .. math::
        \beta(x, x') = a^2 \exp{\frac{(-(x-x')^2 }{(2  \lambda^2))}}
    """

    def fn(
        displacements: np.ndarray[Any, np.dtype[np.float64]],
    ) -> np.ndarray[Any, np.dtype[np.complexfloating]]:
        return (a**2 * np.exp(-(displacements**2) / (2 * sigma**2))).astype(
            np.complex128,
        )

    return fn


def lorentzian_correlation_fn(
    a: float, lambda_: float
) -> Callable[
    [np.ndarray[Any, np.dtype[np.float64]]],
    np.ndarray[Any, np.dtype[np.complexfloating]],
]:
    r"""Get a correlation function for a lorentzian noise kernel.

    A lorentzian noise kernel is isotropic, and separable into individual
    axis kernels. The kernel is given by

    .. math::
        \beta(x, x') = a^2 \frac{\lambda^2}{(x-x')^2 + \lambda^2}
    """

    def fn(
        displacements: np.ndarray[Any, np.dtype[np.float64]],
    ) -> np.ndarray[Any, np.dtype[np.complexfloating]]:
        return (a**2 * lambda_**2 / (displacements**2 + lambda_**2)).astype(
            np.complex128
        )

    return fn


def temperature_corrected_operators[M0: BasisMetadata, M1: BasisMetadata](
    hamiltonian: Operator[M1, np.complexfloating],
    operators: OperatorList[M0, M1, np.complexfloating],
    temperature: float,
    eta: float,
) -> OperatorList[M0, M1, np.complexfloating]:
    """Get the temperature corrected operators.

    Note this returns the operators multiplied by hbar, to avoid numerical issues.
    """
    commutator = get_commutator_operator_list(hamiltonian, operators)
    thermal_energy = Boltzmann * temperature
    correction = commutator * (-1 * np.sqrt(eta / (8 * thermal_energy * hbar**2)))
    operators *= np.sqrt(2 * eta * thermal_energy / hbar**2)
    return correction + operators


def hamiltonian_shift[M1: BasisMetadata](
    hamiltonian: Operator[M1, np.complexfloating],
    operators: OperatorList[BasisMetadata, M1, np.complexfloating],
    eta: float,
) -> Operator[M1, np.complexfloating]:
    """Get the temperature corrected Hamiltonian shift."""
    shift_product = linalg.einsum(
        "(i (j k)),(i (k' l))->(k' l)", operator.dagger_each(operators), operators
    )
    shift_product = Operator(shift_product.basis, shift_product.raw_data)
    commutator = operator.commute(hamiltonian, shift_product)
    pre_factor = 1j * eta / (4 * hbar)
    return commutator * pre_factor


def truncate_noise_operator_list[M0: EigenvalueMetadata, M1: BasisMetadata](
    operators: OperatorList[M0, M1, np.complexfloating],
    truncation: Iterable[int],
) -> OperatorList[M0, M1, np.complexfloating]:
    """
    Get a truncated list of diagonal operators.

    Parameters
    ----------
    operators : DiagonalNoiseOperatorList[FundamentalBasis[BasisMetadata], B_0, B_1]
    truncation : Iterable[int]

    Returns
    -------
    DiagonalNoiseOperatorList[FundamentalBasis[BasisMetadata], B_0, B_1]
    """
    converted = operators.with_basis(basis.as_tuple(operators.basis))
    converted_list = converted.with_list_basis(basis.as_index(converted.basis[0]))
    eigenvalues = (
        converted_list.basis.metadata()
        .children[0]
        .values[converted_list.basis[0].points]
    )
    args = np.argsort(np.abs(eigenvalues))[::-1][np.array(list(truncation))]
    list_basis = CoordinateBasis(args, basis.as_tuple(operators.basis)[0])
    return operators.with_list_basis(list_basis)


def truncate_noise_kernel[M: SuperOperatorMetadata](
    kernel: NoiseKernel[M, np.complexfloating],
    truncation: Iterable[int],
) -> NoiseKernel[M, np.complexfloating]:
    """
    Given a noise kernel, retain only the first n noise operators.

    Parameters
    ----------
    kernel : NoiseKernel[B_0, B_1, B_0, B_1]
    n : int

    Returns
    -------
    NoiseKernel[B_0, B_1, B_0, B_1]
    """
    operators = get_periodic_noise_operators_eigenvalue(kernel)

    truncated = truncate_noise_operator_list(operators, truncation)
    return NoiseKernel.from_operators(truncated)


def truncate_diagonal_noise_kernel[
    M: BasisMetadata,
](
    kernel: DiagonalNoiseKernel[M, np.complexfloating],
    truncation: Iterable[int],
) -> DiagonalNoiseKernel[M, np.complexfloating]:
    """
    Given a noise kernel, retain only the first n noise operators.

    Parameters
    ----------
    kernel : NoiseKernel[B_0, B_1, B_0, B_1]
    n : int

    Returns
    -------
    NoiseKernel[B_0, B_1, B_0, B_1]
    """
    operators = get_periodic_noise_operators_diagonal_eigenvalue(kernel)

    truncated = truncate_noise_operator_list(operators, truncation)
    return DiagonalNoiseKernel.from_operators(truncated)
