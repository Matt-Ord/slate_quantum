from __future__ import annotations

from typing import Any, Self

import numpy as np
from slate.array import SlateArray
from slate.basis import Basis
from slate.basis.stacked import TupleBasis
from slate.metadata import BasisMetadata
from slate.util import slice_ignoring_axes

from slate_quantum.model.operator._operator import Operator

NoiseOperator = Operator


class NoiseKernel[DT: np.generic, B: Basis[BasisMetadata, Any]](SlateArray[DT, B]):
    r"""
    Represents a noise kernel which is isotropic.

    In this case, the correllation between any pair of states depends only on
    the difference between the two states. We therefore store the kernel
    relating to only a single state.
    """

    def with_basis[B1: Basis[Any, Any]](  # B1: B
        self: Self, basis: B1
    ) -> NoiseKernel[DT, B1]:
        """Get the SlateArray with the basis set to basis."""
        return NoiseKernel(
            basis, self.basis.__convert_vector_into__(self.raw_data, basis)
        )


class IsotropicNoiseKernel[DT: np.generic, B: Basis[BasisMetadata, Any]](
    SlateArray[DT, B]
):
    r"""
    Represents a noise kernel which is isotropic.

    In this case, the correllation between any pair of states depends only on
    the difference between the two states. We therefore store the kernel
    relating to only a single state.

    This should therefore be represented using a RecastBasis.
    """

    def with_outer_basis[B1: Basis[BasisMetadata, Any]](
        self, basis: B1
    ) -> IsotropicNoiseKernel[DT, B1]:
        """Get the Potential with the outer recast basis set to basis."""
        return IsotropicNoiseKernel(
            basis, self.basis.__convert_vector_into__(self.raw_data, basis)
        )


type AxisKernel[DT: np.complex128, B: Basis[BasisMetadata, Any]] = tuple[
    IsotropicNoiseKernel[DT, B],
    ...,
]


def _outer_product(
    *arrays: np.ndarray[Any, np.dtype[np.complex128]],
) -> np.ndarray[Any, np.dtype[np.complex128]]:
    grids = np.meshgrid(*arrays, indexing="ij")
    return np.prod(grids, axis=0)


def as_isotropic_kernel_from_axis[DT: np.complex128, M: BasisMetadata](
    kernels: AxisKernel[DT, Basis[M, Any]],
) -> IsotropicNoiseKernel[np.complex128, TupleBasis[M, None, DT]]:
    """Convert an axis kernel to an isotropic kernel."""
    full_basis = tuple(kernel_i.basis for kernel_i in kernels)
    full_data = tuple(kernel_i.raw_data.ravel() for kernel_i in kernels)

    return IsotropicNoiseKernel(
        TupleBasis(full_basis, None), _outer_product(*full_data).ravel()
    )


def as_axis_kernel_from_isotropic[DT: np.complex128, M: BasisMetadata](
    kernels: IsotropicNoiseKernel[np.complex128, TupleBasis[M, None, DT]],
) -> AxisKernel[DT, Basis[M, Any]]:
    """Convert an isotropic kernel to an axis kernel."""
    n_axis = kernels.basis.n_dim

    data_stacked = kernels.raw_data.reshape(kernels.basis.shape)
    slice_without_idx = tuple(0 for _ in range(n_axis - 1))

    prefactor = kernels.raw_data[0] ** ((1 - n_axis) / n_axis)
    return tuple(
        IsotropicNoiseKernel(
            axis_basis,
            prefactor * data_stacked[slice_ignoring_axes(slice_without_idx, (i,))],
        )
        for i, axis_basis in enumerate(tuple(kernels.basis[i] for i in range(n_axis)))
    )
