from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.constants import Boltzmann, hbar  # type: ignore stubs
from slate_core import Basis, TupleMetadata, basis, linalg
from slate_core.basis import CoordinateBasis
from slate_core.metadata import AxisDirections

from slate_quantum import operator
from slate_quantum.metadata import EigenvalueMetadata
from slate_quantum.noise._kernel import (
    DiagonalNoiseKernelWithMetadata,
    IsotropicNoiseKernelWithMetadata,
    NoiseKernelWithMetadata,
    build_isotropic_kernel,
    diagonal_kernel_from_operators,
    noise_kernel_from_operators,
)
from slate_quantum.noise.diagonalize import (
    get_periodic_noise_operators_diagonal_eigenvalue,
    get_periodic_noise_operators_eigenvalue,
)
from slate_quantum.operator import (
    SuperOperatorMetadata,
    get_commutator_operator_list,
)
from slate_quantum.operator._operator import Operator

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from slate_core.metadata import (
        BasisMetadata,
    )
    from slate_core.metadata.length import LengthMetadata, SpacedLengthMetadata

    from slate_quantum.noise._kernel import AxisKernel
    from slate_quantum.operator._operator import (
        OperatorList,
        OperatorListWithMetadata,
        OperatorWithMetadata,
    )


def isotropic_kernel_from_function[M: LengthMetadata](
    metadata: M,
    fn: Callable[
        [np.ndarray[Any, np.dtype[np.float64]]],
        np.ndarray[Any, np.dtype[np.complexfloating]],
    ],
) -> IsotropicNoiseKernelWithMetadata[M, np.dtype[np.complexfloating]]:
    """Get an Isotropic Kernel with a correlation beta(x-x')."""
    displacements = operator.build.x_displacement_operator(metadata)
    correlation = fn(displacements.raw_data.reshape(displacements.basis.inner.shape)[0])

    return build_isotropic_kernel(displacements.basis.inner.children[0], correlation)


def isotropic_kernel_from_function_stacked[
    M: SpacedLengthMetadata,
    E: AxisDirections,
](
    metadata: TupleMetadata[tuple[M, ...], E],
    fn: Callable[
        [np.ndarray[Any, np.dtype[np.float64]]],
        np.ndarray[Any, np.dtype[np.complexfloating]],
    ],
) -> IsotropicNoiseKernelWithMetadata[
    TupleMetadata[tuple[M, ...], E], np.dtype[np.complexfloating]
]:
    """Get an Isotropic Kernel with a correlation beta(x-x')."""
    displacements = operator.build.total_x_displacement_operator(metadata)
    correlation = fn(displacements.raw_data.reshape(displacements.basis.inner.shape)[0])

    return build_isotropic_kernel(displacements.basis.inner.children[0], correlation)


def axis_kernel_from_function_stacked[M: SpacedLengthMetadata](
    metadata: TupleMetadata[tuple[M, ...], Any],
    fn: Callable[
        [np.ndarray[Any, np.dtype[np.float64]]],
        np.ndarray[Any, np.dtype[np.complexfloating]],
    ],
) -> AxisKernel[M, np.complexfloating]:
    """Get an Isotropic Kernel with a correlation beta(x-x')."""
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
    hamiltonian: OperatorWithMetadata[M1, np.dtype[np.complexfloating]],
    operators: OperatorListWithMetadata[M0, M1, np.dtype[np.complexfloating]],
    temperature: float,
    eta: float,
) -> OperatorListWithMetadata[M0, M1, np.dtype[np.complexfloating]]:
    """Get the temperature corrected operators.

    Note this returns the operators multiplied by hbar, to avoid numerical issues.
    """
    commutator = get_commutator_operator_list(hamiltonian, operators)
    thermal_energy = Boltzmann * temperature
    correction = commutator * (-1 * np.sqrt(eta / (8 * thermal_energy * hbar**2)))
    operators *= np.sqrt(2 * eta * thermal_energy / hbar**2)
    return correction + operators


def hamiltonian_shift[M1: BasisMetadata](
    hamiltonian: OperatorWithMetadata[M1, np.dtype[np.complexfloating]],
    operators: OperatorListWithMetadata[
        BasisMetadata, M1, np.dtype[np.complexfloating]
    ],
    eta: float,
) -> OperatorWithMetadata[M1, np.dtype[np.complexfloating]]:
    """Get the temperature corrected Hamiltonian shift."""
    shift_product = linalg.einsum(
        "(i (j k)),(i (k' l))->(k' l)", operator.dagger_each(operators), operators
    )
    shift_product = Operator(shift_product.basis, shift_product.raw_data)
    commutator = operator.commute(hamiltonian, shift_product)
    pre_factor = 1j * eta / (4 * hbar)
    return (commutator * pre_factor).as_type(np.complex128)


def truncate_noise_operator_list[M0: EigenvalueMetadata, M1: BasisMetadata](
    operators: OperatorList[
        Basis[TupleMetadata[tuple[M0, TupleMetadata[tuple[M1, M1], None]], None]],
        np.dtype[np.complexfloating],
    ],
    truncation: Iterable[int],
) -> OperatorList[
    Basis[TupleMetadata[tuple[M0, TupleMetadata[tuple[M1, M1], None]], None]],
    np.dtype[np.complexfloating],
]:
    """Get a truncated list of diagonal operators."""
    converted = operators.with_basis(basis.as_tuple(operators.basis).upcast())
    converted_list = converted.with_list_basis(
        basis.as_index(converted.basis.inner.children[0])
    )
    eigenvalues = (
        converted_list.basis.metadata()
        .children[0]
        .values[converted_list.basis.inner.children[0].points]
    )
    args = np.argsort(np.abs(eigenvalues))[::-1][np.array(list(truncation))]
    list_basis = CoordinateBasis(args, basis.as_tuple(operators.basis).children[0])
    return operators.with_list_basis(list_basis)


def truncate_noise_kernel[M: SuperOperatorMetadata](
    kernel: NoiseKernelWithMetadata[M, np.dtype[np.complexfloating]],
    truncation: Iterable[int],
) -> NoiseKernelWithMetadata[M, np.dtype[np.complexfloating]]:
    """Given a noise kernel, retain only the first n noise operators."""
    operators = get_periodic_noise_operators_eigenvalue(kernel)

    truncated = truncate_noise_operator_list(operators, truncation)
    return noise_kernel_from_operators(truncated)


def truncate_diagonal_noise_kernel[
    M: BasisMetadata,
](
    kernel: DiagonalNoiseKernelWithMetadata[M, np.dtype[np.complexfloating]],
    truncation: Iterable[int],
) -> DiagonalNoiseKernelWithMetadata[M, np.dtype[np.complexfloating]]:
    """Given a noise kernel, retain only the first n noise operators."""
    operators = get_periodic_noise_operators_diagonal_eigenvalue(kernel)

    truncated = truncate_noise_operator_list(operators, truncation)
    return diagonal_kernel_from_operators(truncated)
