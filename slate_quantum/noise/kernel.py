from __future__ import annotations

from typing import Any, Self, cast

import numpy as np
from slate.array import SlateArray
from slate.basis import Basis, FundamentalBasis, RecastBasis
from slate.basis.stacked import as_tuple_basis, isotropic_basis, tuple_basis
from slate.metadata import BasisMetadata, StackedMetadata
from slate.util import slice_ignoring_axes


class DiagonalNoiseKernel[
    M: BasisMetadata,
    DT: np.generic,
    B: Basis[BasisMetadata, Any] = Basis[M, DT],
](SlateArray[M, DT, B]):
    r"""Represents a noise kernel which is diagonal."""

    def __init__[DT1: np.generic, B1: Basis[BasisMetadata, Any]](
        self: DiagonalNoiseKernel[Any, DT1, B1],
        basis: B1,
        data: np.ndarray[Any, np.dtype[DT]],
    ) -> None:
        super().__init__(cast(Any, basis), cast(Any, data))

    def with_basis[B1: Basis[Any, Any]](  # B1: B
        self: Self, basis: B1
    ) -> DiagonalNoiseKernel[M, DT, B1]:
        """Get the SlateArray with the basis set to basis."""
        return DiagonalNoiseKernel(
            basis, self.basis.__convert_vector_into__(self.raw_data, basis)
        )


type IsotropicBasis[M: BasisMetadata, DT: np.generic] = RecastBasis[
    StackedMetadata[M, None], M, DT
]


class IsotropicNoiseKernel[
    M: BasisMetadata,
    DT: np.generic,
](DiagonalNoiseKernel[M, DT, IsotropicBasis[M, DT]]):
    r"""
    Represents a noise kernel which is isotropic.

    In this case, the correllation between any pair of states depends only on
    the difference between the two states. We therefore store the kernel
    relating to only a single state.

    This should therefore be represented using a RecastBasis.
    """

    def __init__[
        M1: BasisMetadata,
        DT1: np.generic,
    ](
        self: IsotropicNoiseKernel[M1, DT1],
        basis: Basis[M1, DT1],
        data: np.ndarray[Any, np.dtype[DT]],
    ) -> None:
        fundamental = cast(Basis[Any, Any], FundamentalBasis(basis.metadata))
        recast = RecastBasis(
            isotropic_basis((fundamental, fundamental.conjugate_basis())),
            fundamental,
            basis,
        )
        super().__init__(cast(Any, recast), cast(Any, data))

    def with_outer_basis[M1: BasisMetadata](
        self, basis: Basis[M1, Any]
    ) -> IsotropicNoiseKernel[M1, DT]:
        """Get the Potential with the outer recast basis set to basis."""
        return IsotropicNoiseKernel(
            basis, self.basis.outer_recast.__convert_vector_into__(self.raw_data, basis)
        )


type AxisKernel[M: BasisMetadata, DT: np.complex128] = tuple[
    IsotropicNoiseKernel[M, DT],
    ...,
]


def _outer_product(
    *arrays: np.ndarray[Any, np.dtype[np.complex128]],
) -> np.ndarray[Any, np.dtype[np.complex128]]:
    grids = np.meshgrid(*arrays, indexing="ij")
    return np.prod(grids, axis=0)


def as_isotropic_kernel_from_axis[M: BasisMetadata, DT: np.complex128](
    kernels: AxisKernel[M, DT],
) -> IsotropicNoiseKernel[StackedMetadata[M, None], np.complex128]:
    """Convert an axis kernel to an isotropic kernel."""
    full_basis = tuple(kernel_i.basis for kernel_i in kernels)
    full_data = tuple(kernel_i.raw_data.ravel() for kernel_i in kernels)

    return IsotropicNoiseKernel(
        tuple_basis(full_basis, None), _outer_product(*full_data).ravel()
    )


def as_axis_kernel_from_isotropic[
    M: BasisMetadata,
    DT: np.complex128,
](
    kernel: IsotropicNoiseKernel[StackedMetadata[M, Any], DT],
) -> AxisKernel[M, DT]:
    """Convert an isotropic kernel to an axis kernel."""
    outer_as_tuple = as_tuple_basis(kernel.basis.outer_recast)
    converted = kernel.with_outer_basis(outer_as_tuple)
    n_axis = converted.basis.n_dim

    data_stacked = converted.raw_data.reshape(outer_as_tuple.shape)
    slice_without_idx = tuple(0 for _ in range(n_axis - 1))

    prefactor = converted.raw_data[0] ** ((1 - n_axis) / n_axis)
    return tuple(
        IsotropicNoiseKernel(
            axis_basis,
            prefactor * data_stacked[slice_ignoring_axes(slice_without_idx, (i,))],
        )
        for i, axis_basis in enumerate(outer_as_tuple.children)
    )
