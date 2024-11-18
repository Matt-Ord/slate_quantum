from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from scipy.constants import Boltzmann  # type: ignore stubs

from slate_quantum.model.operator.build._displacement import (
    build_total_x_displacement_operator,
    build_x_displacement_operator,
)
from slate_quantum.model.operator.linalg import get_commutator_operator_list
from slate_quantum.noise.kernel import IsotropicNoiseKernel

if TYPE_CHECKING:
    import numpy as np
    from slate.metadata import BasisMetadata
    from slate.metadata.length import LengthMetadata, SpacedLengthMetadata
    from slate.metadata.stacked import SpacedVolumeMetadata, StackedMetadata

    from slate_quantum.model.operator import (
        Operator,
        OperatorList,
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


def get_temperature_corrected_operators[M: StackedMetadata[BasisMetadata, Any]](
    hamiltonian: Operator[Any, np.complex128],
    operators: OperatorList[M, np.complex128],
    temperature: float,
) -> OperatorList[M, np.complex128]:
    """Get the temperature corrected operators."""
    commutator = get_commutator_operator_list(hamiltonian, operators)
    correction = commutator * (-1 / (4 * Boltzmann * temperature))
    return correction + operators
