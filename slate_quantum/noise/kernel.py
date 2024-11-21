from __future__ import annotations

from typing import Any, cast

import numpy as np
from slate.basis import Basis, DiagonalBasis, FundamentalBasis, RecastBasis
from slate.basis.stacked import (
    TupleBasis2D,
    as_tuple_basis,
    diagonal_basis,
    isotropic_basis,
    tuple_basis,
)
from slate.basis.wrapped import as_index_basis
from slate.metadata import BasisMetadata, Metadata2D, StackedMetadata
from slate.util import slice_ignoring_axes

from slate_quantum._util import outer_product
from slate_quantum.model._label import EigenvalueMetadata
from slate_quantum.model.operator._operator import OperatorList
from slate_quantum.model.operator._super_operator import (
    SuperOperator,
    SuperOperatorMetadata,
)

_NoiseKernelMetadata = SuperOperatorMetadata[
    BasisMetadata, BasisMetadata, BasisMetadata, BasisMetadata
]


class NoiseKernel[
    M: _NoiseKernelMetadata,
    DT: np.generic,
    B: Basis[_NoiseKernelMetadata, Any] = Basis[M, DT],
](SuperOperator[M, DT, B]):
    r"""Represents a noise kernel which is diagonal."""

    def __init__[
        DT1: np.generic,
        B1: Basis[_NoiseKernelMetadata, Any],
    ](
        self: NoiseKernel[Any, DT1, B1],
        basis: B1,
        data: np.ndarray[Any, np.dtype[DT]],
    ) -> None:
        super().__init__(cast(Any, basis), cast(Any, data))


type DiagonalKernelMetadata[M0: BasisMetadata, M1: BasisMetadata] = Metadata2D[
    Metadata2D[M0, M0, Any], Metadata2D[M1, M1, Any], None
]

type DiagonalKernelBasis[
    M0: BasisMetadata,
    M1: BasisMetadata,
    DT: np.generic,
] = RecastBasis[
    DiagonalKernelMetadata[M0, M1],
    Metadata2D[M0, M1, None],
    DT,
]


class DiagonalNoiseKernel[
    M0: BasisMetadata,
    M1: BasisMetadata,
    DT: np.generic,
    B: Basis[
        DiagonalKernelMetadata[BasisMetadata, BasisMetadata], Any
    ] = DiagonalKernelBasis[M0, M1, DT],
](
    NoiseKernel[
        Metadata2D[Metadata2D[M0, M0, Any], Metadata2D[M0, M0, Any], None], DT, B
    ]
):
    r"""Represents a noise kernel which is diagonal.

    If a kernel with basis ((a, b), (c, d)) is diagonal, then the kernel is
    represented as a 2D array indexed ((a, a), (b, b)).

    In this case, we can re-cast the kernel in the basis (a, b) and store
    only the diagonal elements.

    DiagonalNoiseKernel(basis, data) creates a DiagonalNoiseKernel with the
    outer_basis set to basis and the data set to data.

    The inner_basis is set to as_tuple_basis(basis).
    """

    def __init__[
        _M0: BasisMetadata,
        _M1: BasisMetadata,
        _DT: np.generic,
        _B: Basis[
            DiagonalKernelMetadata[BasisMetadata, BasisMetadata], Any
        ] = DiagonalKernelBasis[_M0, _M1, _DT],
    ](
        self: DiagonalNoiseKernel[_M0, _M1, _DT, _B],
        basis: Basis[Metadata2D[M0, M1, Any], Any],
        data: np.ndarray[Any, np.dtype[DT]],
    ) -> None:
        outer_recast = basis
        inner_recast = as_tuple_basis(basis)
        b0 = inner_recast.children[0]
        b1 = inner_recast.children[1]
        inner = tuple_basis((diagonal_basis((b0, b0)), diagonal_basis((b1, b1))))
        actual_basis = RecastBasis(inner, inner_recast, outer_recast)
        super().__init__(cast(Any, actual_basis), cast(Any, data))

    def with_outer_basis[_M0: BasisMetadata, _M1: BasisMetadata, _DT: np.generic](
        self: DiagonalNoiseKernel[_M0, _M1, _DT],
        basis: Basis[Metadata2D[_M0, _M1, Any], Any],
    ) -> DiagonalNoiseKernel[_M0, _M1, _DT]:
        """Get the Potential with the outer recast basis set to basis."""
        return DiagonalNoiseKernel[_M0, _M1, _DT](
            basis,
            self.basis.outer_recast.__convert_vector_into__(self.raw_data, basis),
        )


type IsotropicKernelMetadata[M: BasisMetadata] = DiagonalKernelMetadata[M, M]
type IsotropicBasis[M: BasisMetadata, DT: np.generic] = RecastBasis[
    IsotropicKernelMetadata[M], M, DT, DiagonalKernelBasis[M, M, DT]
]


class IsotropicNoiseKernel[
    M: BasisMetadata,
    DT: np.generic,
    B: Basis[IsotropicKernelMetadata[BasisMetadata], Any] = IsotropicBasis[M, DT],
](DiagonalNoiseKernel[M, M, DT, B]):
    r"""
    Represents a noise kernel which is isotropic and diagonal.

    In this case, the correllation between any pair of states depends only on
    the difference between the two states. We therefore store the kernel
    relating to only a single state.

    The full kernel has the basis ((a, a), (a, a)), and the isotropic kernel
    is indexed according to the basis ((a, a), (0, 0)).
    """

    def __init__[
        _M: BasisMetadata,
        _DT: np.generic,
    ](
        self: IsotropicNoiseKernel[_M, _DT, IsotropicBasis[_M, _DT]],
        basis: Basis[_M, _DT],
        data: np.ndarray[Any, np.dtype[DT]],
    ) -> None:
        outer_recast = basis
        inner_recast = basis
        inner = isotropic_basis((basis, basis.conjugate_basis()))

        recast = RecastBasis(inner, inner_recast, outer_recast)
        super().__init__(cast(Any, recast), cast(Any, data))

    def with_isotropic_basis[
        _M: BasisMetadata,
        _DT: np.generic,
    ](
        self: IsotropicNoiseKernel[
            _M,
            _DT,
        ],
        basis: Basis[BasisMetadata, Any],
    ) -> IsotropicNoiseKernel[_M, _DT]:
        """Get the Potential with the outer recast basis set to basis."""
        return IsotropicNoiseKernel(
            cast(Basis[_M, Any], basis),
            self.basis.outer_recast.__convert_vector_into__(self.raw_data, basis),
        )

    def unwrap[
        _M: BasisMetadata,
        _DT: np.generic,
    ](
        self: IsotropicNoiseKernel[
            _M,
            _DT,
        ],
    ) -> DiagonalNoiseKernel[_M, _M, _DT]:
        """Unwrap the isotropic kernel into a diagonal kernel."""
        return DiagonalNoiseKernel[_M, _M, _DT, Any](
            self.basis.inner.outer_recast,
            self.basis.__convert_vector_into__(self.raw_data, self.basis.inner),
        )


type AxisKernel[M: BasisMetadata, DT: np.complex128] = tuple[
    IsotropicNoiseKernel[M, DT],
    ...,
]


def as_isotropic_kernel_from_axis[M: BasisMetadata, DT: np.complex128](
    kernels: AxisKernel[M, DT],
) -> IsotropicNoiseKernel[StackedMetadata[M, Any], np.complex128]:
    """Convert an axis kernel to an isotropic kernel."""
    full_basis = tuple(kernel_i.basis for kernel_i in kernels)
    full_data = tuple(kernel_i.raw_data.ravel() for kernel_i in kernels)

    return IsotropicNoiseKernel(
        tuple_basis(full_basis, None), outer_product(*full_data).ravel()
    )


def as_axis_kernel_from_isotropic[
    M: BasisMetadata,
    DT: np.complex128,
](
    kernel: IsotropicNoiseKernel[StackedMetadata[M, Any], DT],
) -> AxisKernel[M, DT]:
    """Convert an isotropic kernel to an axis kernel."""
    outer_as_tuple = as_tuple_basis(kernel.basis.outer_recast)
    converted = kernel.with_isotropic_basis(outer_as_tuple)
    n_axis = converted.basis.n_dim

    data_stacked = converted.raw_data.reshape(outer_as_tuple.shape)
    slice_without_idx = tuple(0 for _ in range(n_axis - 1))

    prefactor = converted.raw_data[0] ** ((1 - n_axis) / n_axis)
    return tuple(
        IsotropicNoiseKernel[M, DT](
            axis_basis,
            prefactor * data_stacked[slice_ignoring_axes(slice_without_idx, (i,))],
        )
        for i, axis_basis in enumerate(outer_as_tuple.children)
    )


def get_full_kernel_from_operators[M0: BasisMetadata, M1: BasisMetadata](
    operators: OperatorList[
        Metadata2D[EigenvalueMetadata, Metadata2D[M0, M1, Any], Any],
        np.complex128,
    ],
) -> NoiseKernel[
    Any,
    np.complex128,
    TupleBasis2D[
        Any,
        TupleBasis2D[np.generic, Basis[M0, Any], Basis[M1, Any], Any],
        TupleBasis2D[np.generic, Basis[M0, Any], Basis[M1, Any], Any],
        None,
    ],
]:
    """
    Build a full kernel from operators.

    Parameters
    ----------
    operators : NoiseOperatorList[FundamentalBasis[BasisMetadata], _B0, _B1]

    Returns
    -------
    NoiseKernel[_B0, _B1, _B0, _B1]
    """
    converted = operators.with_basis(as_tuple_basis(operators.basis))
    converted_inner = converted.with_operator_basis(
        as_tuple_basis(converted.basis[1])
    ).with_list_basis(as_index_basis(converted.basis[0]))
    operators_data = converted_inner.raw_data.reshape(
        converted.basis[0].size, *converted_inner.basis[1].shape
    )

    data = np.einsum(  # type:ignore  unknown
        "a,aji,akl->ij kl",
        converted.basis[0].metadata().values[converted_inner.basis.points],
        np.conj(operators_data),
        operators_data,
    )
    return NoiseKernel(
        tuple_basis((converted_inner.basis[1], converted_inner.basis[1])), data
    )


def get_diagonal_kernel_from_operators[
    M0: BasisMetadata,
    M1: BasisMetadata,
](
    operators: OperatorList[
        Metadata2D[EigenvalueMetadata, Metadata2D[M0, M1, Any], Any],
        np.complex128,
    ],
) -> DiagonalNoiseKernel[M0, M1, np.complex128]:
    """Build a diagonal kernel from operators."""
    converted = operators.with_basis(as_tuple_basis(operators.basis))
    converted_inner = converted.with_operator_basis(
        DiagonalBasis(as_tuple_basis(converted.basis[1]))
    ).with_list_basis(as_index_basis(converted.basis[0]))

    operators_data = converted_inner.raw_data.reshape(converted.basis[0].size, -1)
    data = np.einsum(  # type:ignore  unknown
        "a,ai,aj->ij",
        converted.basis[0].metadata().values[converted_inner.basis[0].points],
        np.conj(operators_data),
        operators_data,
    )
    return DiagonalNoiseKernel[M0, M1, np.complex128](
        converted_inner.basis[1].inner,
        data,
    )


def get_diagonal_noise_operators_from_axis[M: BasisMetadata, E](
    operators_list: tuple[
        OperatorList[
            Metadata2D[EigenvalueMetadata, Metadata2D[M, M, Any], Any],
            np.complex128,
        ],
        ...,
    ],
    extra: E,
) -> OperatorList[
    Metadata2D[
        EigenvalueMetadata,
        Metadata2D[
            StackedMetadata[M, E],
            StackedMetadata[M, E],
            None,
        ],
        Any,
    ],
    np.complex128,
    TupleBasis2D[
        np.complex128,
        FundamentalBasis[EigenvalueMetadata],
        DiagonalBasis[
            np.complex128,
            Basis[StackedMetadata[M, E], Any],
            Basis[StackedMetadata[M, E], Any],
            None,
        ],
        None,
    ],
]:
    """Convert axis operators into full operators."""
    op_as_tuple = tuple(
        operators.with_basis(as_tuple_basis(operators.basis))
        for operators in operators_list
    )
    op_as_tuple_nested = tuple(
        operators.with_list_basis(
            as_index_basis(operators.basis[0])
        ).with_operator_basis(DiagonalBasis(as_tuple_basis(operators.basis[1])))
        for operators in op_as_tuple
    )

    full_basis_0 = tuple_basis(
        tuple(operators.basis[1].inner[0] for operators in op_as_tuple_nested), extra
    )
    full_basis_1 = tuple_basis(
        tuple(operators.basis[1].inner[1] for operators in op_as_tuple_nested), extra
    )

    # for example, in 2d this is ij,kl -> ikjl
    subscripts = tuple(
        (chr(ord("i") + i), chr(ord("i") + i + 1))
        for i in range(0, len(operators_list) * 2, 2)
    )
    input_subscripts = ",".join(["".join(group) for group in subscripts])
    output_subscript = "".join("".join(group) for group in zip(*subscripts))
    einsum_string = f"{input_subscripts}->{output_subscript}"

    full_data = tuple(
        operators.raw_data.reshape(operators.basis[0].size, -1)
        for operators in op_as_tuple
    )
    data = cast(np.ndarray[Any, Any], np.einsum(einsum_string, *full_data))  # type: ignore unknown
    full_coefficients = tuple(
        operators.basis[0].metadata().values[operators.basis[0].points]
        for operators in op_as_tuple
    )
    eigenvalues = outer_product(*full_coefficients)
    eigenvalue_basis = FundamentalBasis(EigenvalueMetadata(eigenvalues))

    return OperatorList(
        tuple_basis((eigenvalue_basis, diagonal_basis((full_basis_0, full_basis_1)))),
        data,
    )
