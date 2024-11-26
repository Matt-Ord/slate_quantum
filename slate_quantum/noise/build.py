from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, cast

import numpy as np
from scipy.constants import Boltzmann  # type: ignore stubs
from slate.basis import CoordinateBasis, as_index_basis, as_tuple_basis

from slate_quantum.model._label import EigenvalueMetadata
from slate_quantum.model.operator import (
    OperatorList,
)
from slate_quantum.model.operator._super_operator import SuperOperatorMetadata
from slate_quantum.model.operator.build._position import (
    build_total_x_displacement_operator,
    build_x_displacement_operator,
)
from slate_quantum.model.operator.linalg import get_commutator_operator_list
from slate_quantum.noise.diagonalize._eigenvalue import (
    get_periodic_noise_operators_diagonal_eigenvalue,
    get_periodic_noise_operators_eigenvalue,
)
from slate_quantum.noise.kernel import (
    DiagonalNoiseKernel,
    IsotropicNoiseKernel,
    NoiseKernel,
    get_diagonal_kernel_from_operators,
    get_full_kernel_from_operators,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from slate.metadata import (
        BasisMetadata,
        Metadata2D,
        SpacedVolumeMetadata,
        StackedMetadata,
    )
    from slate.metadata.length import LengthMetadata, SpacedLengthMetadata

    from slate_quantum.model.operator import (
        Operator,
    )

    from .kernel import AxisKernel


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
    displacements = build_x_displacement_operator(metadata)
    correlation = fn(displacements.raw_data.reshape(displacements.basis.shape)[0])

    return IsotropicNoiseKernel(displacements.basis[0], correlation)


def build_isotropic_kernel_from_function_stacked[M: SpacedVolumeMetadata](
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
    displacements = build_total_x_displacement_operator(metadata)
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


def get_temperature_corrected_operators[
    M: Metadata2D[BasisMetadata, BasisMetadata, Any]
](
    hamiltonian: Operator[Any, np.complex128],
    operators: OperatorList[M, np.complex128],
    temperature: float,
) -> OperatorList[M, np.complex128]:
    """Get the temperature corrected operators."""
    commutator = get_commutator_operator_list(hamiltonian, operators)
    correction = commutator * (-1 / (4 * Boltzmann * temperature))
    return correction + operators


def truncate_noise_operator_list[M: Metadata2D[EigenvalueMetadata, BasisMetadata, Any]](
    operators: OperatorList[M, np.complex128],
    truncation: Iterable[int],
) -> OperatorList[M, np.complex128]:
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
    return cast(OperatorList[M, np.complex128], operators.with_list_basis(list_basis))


def truncate_noise_kernel[
    M: SuperOperatorMetadata[
        BasisMetadata, BasisMetadata, BasisMetadata, BasisMetadata
    ],
](
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
    return cast(
        NoiseKernel[M, np.complex128], get_full_kernel_from_operators(truncated)
    )


def truncate_diagonal_noise_kernel[
    M0: BasisMetadata,
    M1: BasisMetadata,
](
    kernel: DiagonalNoiseKernel[M0, M1, np.complex128],
    truncation: Iterable[int],
) -> DiagonalNoiseKernel[M0, M1, np.complex128]:
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
    return get_diagonal_kernel_from_operators(truncated)
