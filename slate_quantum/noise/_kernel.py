from __future__ import annotations

from typing import Any, cast

import numpy as np
from slate_core import Ctype, TupleBasis
from slate_core import basis as basis_
from slate_core.basis import (
    Basis,
    DiagonalBasis,
    FundamentalBasis,
    IsotropicBasis,
    RecastBasis,
)
from slate_core.metadata import (
    BasisMetadata,
)
from slate_core.util import slice_ignoring_axes

from slate_quantum._util import outer_product
from slate_quantum.metadata import EigenvalueMetadata
from slate_quantum.noise.legacy import (
    LegacyBasis,
    LegacyDiagonalBasis,
    LegacyIsotropicBasis,
    LegacyRecastBasis,
    LegacyTupleBasis2D,
    Metadata2D,
    StackedMetadata,
    diagonal_basis,
    tuple_basis,
)
from slate_quantum.operator._diagonal import recast_diagonal_basis
from slate_quantum.operator._operator import (
    LegacyOperator,
    LegacyOperatorList,
    Operator,
    OperatorMetadata,
    SuperOperatorMetadata,
    build_legacy_operator_list,
)

type NoiseKernel[
    M: BasisMetadata,
    DT: np.generic,
    B: Basis[SuperOperatorMetadata, Any] = Basis[SuperOperatorMetadata[M], Ctype[DT]],
] = LegacyOperator[M, DT, B]


def noise_kernel_from_operators[M_: BasisMetadata, DT_: np.generic](
    operators: LegacyOperatorList[EigenvalueMetadata, M_, DT_],
) -> NoiseKernel[M_, DT_]:
    converted = operators.with_basis(
        basis_.as_tuple(operators.basis).upcast()
    ).assert_ok()
    converted_inner = (
        converted.with_operator_basis(
            basis_.as_tuple(converted.basis.inner.children[1]).upcast()
        )
        .assert_ok()
        .with_list_basis(basis_.as_index(converted.basis.inner.children[0]))
        .assert_ok()
    )
    operators_data = converted_inner.raw_data.reshape(
        converted.basis.inner.children[0].size,
        *converted_inner.basis.inner.children[1].shape,  # type:ignore refactor
    )

    data = np.einsum(  # type:ignore  unknown
        "a,aji,akl->ij kl",
        converted.basis.inner.children[0]
        .metadata()
        .values[converted_inner.basis.points],
        np.conj(operators_data),
        operators_data.astype(np.complex128),
    )
    return Operator.build(
        tuple_basis(
            (
                converted_inner.basis.inner.children[1],
                converted_inner.basis.inner.children[1],
            )
        ).upcast(),
        data,
    ).assert_ok()


type DiagonalKernelBasis[
    M: BasisMetadata,
    DT: np.generic,
    OuterB: Basis[Any, Any] = LegacyBasis[OperatorMetadata[M], DT],
] = LegacyRecastBasis[
    SuperOperatorMetadata[M],
    Metadata2D[M, M, None],
    DT,
    LegacyTupleBasis2D[
        np.generic,
        LegacyDiagonalBasis[Any, Basis[M, Any], Basis[M, Any], None],
        LegacyDiagonalBasis[Any, Basis[M, Any], Basis[M, Any], None],
        None,
    ],
    OuterB,
]


type DiagonalNoiseKernel[
    M: BasisMetadata,
    DT: np.generic,
    B: Basis[Any, Any] = DiagonalKernelBasis[M, DT],
] = Operator[B, np.dtype[DT]]
r"""Represents a noise kernel which is diagonal.

If a kernel with basis ((a, b), (c, d)) is diagonal, then the kernel is
represented as a 2D array indexed ((a, a), (b, b)).

In this case, we can re-cast the kernel in the basis (a, b) and store
only the diagonal elements.

DiagonalNoiseKernel(basis, data) creates a DiagonalNoiseKernel with the
outer_basis set to basis and the data set to data.

The inner_basis is set to basis.as_tuple(basis).
"""


def build_diagonal_kernel[
    M_: BasisMetadata,
    DT_: np.generic,
](
    basis: LegacyBasis[M_, DT_],
    data: np.ndarray[Any, np.dtype[DT_]],
) -> DiagonalNoiseKernel[M_, DT_, DiagonalKernelBasis[M_, DT_]]:
    """Build a diagonal kernel."""
    outer_recast = basis
    inner_recast = basis_.as_tuple(basis)
    b0 = inner_recast.children[0]
    b1 = inner_recast.children[1]
    inner = tuple_basis((diagonal_basis((b0, b0)), diagonal_basis((b1, b1))))
    recast = RecastBasis(inner, inner_recast, outer_recast)
    return Operator.build(cast("Any", recast), cast("Any", data)).assert_ok()


def diagonal_kernel_with_outer_basis[
    M_: BasisMetadata,
    DT_: np.generic,
](
    kernel: DiagonalNoiseKernel[M_, DT_],
    basis: Basis[Metadata2D[M_, M_, Any], Any],
) -> DiagonalNoiseKernel[M_, DT_]:
    """Get the Potential with the outer recast basis set to basis."""
    return build_diagonal_kernel(
        basis,
        kernel.basis.outer_recast.__convert_vector_into__(kernel.raw_data, basis),
    )


def diagonal_kernel_from_operators[M_: BasisMetadata, DT_: np.generic](
    operators: LegacyOperatorList[EigenvalueMetadata, M_, DT_],
) -> DiagonalNoiseKernel[M_, DT_]:
    """Build a diagonal kernel from operators."""
    converted = operators.with_basis(
        basis_.as_tuple(operators.basis).upcast()
    ).assert_ok()
    converted_inner = (  # type:ignore refactor
        converted.with_operator_basis(
            DiagonalBasis(basis_.as_tuple(converted.basis.inner.children[1]))  # type:ignore refactor
        )
        .assert_ok()
        .with_list_basis(basis_.as_index(converted.basis.inner.children[0]))
        .assert_ok()
    )

    operators_data = converted_inner.raw_data.reshape(
        converted.basis.inner.children[0].size, -1
    )
    data = cast(
        "Any",
        np.einsum(  # type:ignore  unknown
            "a,ai,aj->ij",
            converted.basis.metadata()
            .children[0]
            .values[converted_inner.basis.inner.children[0].points],  # type:ignore refactor
            np.conj(operators_data),
            operators_data,  # type:ignore DT not numeric
        ),
    )
    return build_diagonal_kernel(converted_inner.basis.inner.children[1].inner, data)  # type:ignore refactor


type IsotropicKernelBasis[
    M: BasisMetadata,
    DT: np.generic,
    OuterB: Basis[BasisMetadata, Any] = LegacyBasis[M, DT],
] = DiagonalKernelBasis[
    M,
    DT,
    LegacyRecastBasis[
        OperatorMetadata[M],
        M,
        DT,
        LegacyIsotropicBasis[Any, Basis[M, Any], Basis[M, Any], None],
        OuterB,
    ],
]

type IsotropicNoiseKernel[
    M: BasisMetadata,
    DT: np.generic,
    B: Basis[Any, Any] = IsotropicKernelBasis[M, DT],
] = LegacyOperator[M, DT, B]


def build_isotropic_kernel[
    M_: BasisMetadata,
    DT_: np.generic,
](
    basis: LegacyBasis[M_, DT_],
    data: np.ndarray[Any, np.dtype[DT_]],
) -> IsotropicNoiseKernel[M_, DT_, IsotropicKernelBasis[M_, DT_]]:
    outer_recast = basis
    inner_recast = basis
    inner = IsotropicBasis(TupleBasis((basis, basis.dual_basis())))

    recast = RecastBasis(inner, inner_recast, outer_recast)
    return build_diagonal_kernel(cast("Any", recast), cast("Any", data))


def isotropic_kernel_with_isotropic_basis[
    M_: BasisMetadata,
    DT_: np.generic,
](
    kernel: IsotropicNoiseKernel[M_, DT_],
    basis: Basis[M_, Any],
) -> IsotropicNoiseKernel[M_, DT_]:
    """Get the Potential with the outer recast basis set to basis."""
    return build_isotropic_kernel(
        basis,
        kernel.basis.outer_recast.outer_recast.__convert_vector_into__(
            kernel.raw_data, basis
        ),
    )


def isotropic_kernel_from_diagonal_kernel[M_: BasisMetadata, DT_: np.generic](
    kernel: DiagonalNoiseKernel[M_, DT_],
) -> IsotropicNoiseKernel[M_, DT_]:
    """Build a diagonal kernel from operators."""
    basis = basis_.as_tuple(kernel.basis.outer_recast).children[0]  # type:ignore refactor
    converted = diagonal_kernel_with_outer_basis(
        kernel,
        IsotropicBasis(TupleBasis((basis, basis.dual_basis()))),  # type:ignore refactor
    )
    return build_isotropic_kernel(basis, converted.raw_data)  # type:ignore refactor


def diagonal_kernel_from_isotropic_kernel[M_: BasisMetadata, DT_: np.generic](
    kernel: IsotropicNoiseKernel[M_, DT_],
) -> DiagonalNoiseKernel[M_, DT_]:
    """Build a diagonal kernel from operators."""
    return build_diagonal_kernel(
        cast("Any", kernel.basis.outer_recast.inner),  # type:ignore refactor
        kernel.raw_data,
    )


def isotropic_kernel_from_operators[M_: BasisMetadata, DT_: np.generic](
    operators: LegacyOperatorList[EigenvalueMetadata, M_, DT_],
) -> IsotropicNoiseKernel[M_, DT_]:
    """Build an isotropic kernel from operators."""
    diagonal_kernel = diagonal_kernel_from_operators(operators)
    return isotropic_kernel_from_diagonal_kernel(diagonal_kernel)  # type:ignore refactor


type AxisKernel[M: BasisMetadata, DT: np.complexfloating] = tuple[
    IsotropicNoiseKernel[M, DT],
    ...,
]


def as_isotropic_kernel_from_axis[M: BasisMetadata, DT: np.complexfloating](
    kernels: AxisKernel[M, DT],
) -> IsotropicNoiseKernel[StackedMetadata[M, Any], np.complexfloating]:
    """Convert an axis kernel to an isotropic kernel."""
    full_basis = tuple(kernel_i.basis for kernel_i in kernels)
    full_data = tuple(kernel_i.raw_data.ravel() for kernel_i in kernels)

    return build_isotropic_kernel(
        tuple_basis(full_basis, None), outer_product(*full_data).ravel()
    )


def as_axis_kernel_from_isotropic[
    M: BasisMetadata,
    DT: np.complexfloating,
](
    kernel: IsotropicNoiseKernel[StackedMetadata[M, Any], DT],
) -> AxisKernel[M, DT]:
    """Convert an isotropic kernel to an axis kernel."""
    outer_as_tuple = basis_.as_tuple(kernel.basis.outer_recast.outer_recast)  # type:ignore refactor
    converted = isotropic_kernel_with_isotropic_basis(kernel, outer_as_tuple)  # type:ignore refactor
    n_axis = len(outer_as_tuple.shape)  # type:ignore refactor

    data_stacked = converted.raw_data.reshape(outer_as_tuple.shape)  # type:ignore refactor
    slice_without_idx = tuple(0 for _ in range(n_axis - 1))

    prefactor = converted.raw_data[0] ** ((1 - n_axis) / n_axis)
    return tuple(
        build_isotropic_kernel(
            axis_basis,  # type:ignore refactor
            prefactor * data_stacked[slice_ignoring_axes(slice_without_idx, (i,))],
        )
        for i, axis_basis in enumerate(outer_as_tuple.children)  # type:ignore refactor
    )


type LegacyRecastDiagonalOperatorBasis[M, DT] = Any


def get_diagonal_noise_operators_from_axis[M: BasisMetadata, E](
    operators_list: tuple[
        LegacyOperatorList[
            EigenvalueMetadata,
            M,
            np.complexfloating,
        ],
        ...,
    ],
    extra: E,
) -> LegacyOperatorList[
    EigenvalueMetadata,
    StackedMetadata[M, E],
    np.complex128,
    LegacyTupleBasis2D[
        np.complexfloating,
        FundamentalBasis[EigenvalueMetadata],
        LegacyRecastDiagonalOperatorBasis[
            StackedMetadata[M, E],
            np.complexfloating,
        ],
        None,
    ],
]:
    """Convert axis operators into full operators."""
    op_as_tuple = tuple(
        operators.with_basis(basis_.as_tuple(operators.basis))
        for operators in operators_list
    )
    op_as_tuple_nested = tuple(
        operators.with_list_basis(basis_.as_index(operators.basis[0]))
        .assert_ok()
        .with_operator_basis(DiagonalBasis(basis_.as_tuple(operators.basis[1])))
        for operators in op_as_tuple
    )

    full_basis_1 = tuple_basis(
        tuple(operators.basis[1].inner[1] for operators in op_as_tuple_nested), extra
    )

    # for example, in 2d this is ij,kl ->  # cSpell:ignore
    subscripts = tuple(
        (chr(ord("i") + i), chr(ord("i") + i + 1))
        for i in range(0, len(operators_list) * 2, 2)
    )
    input_subscripts = ",".join(["".join(group) for group in subscripts])
    output_subscript = "".join(
        "".join(group) for group in zip(*subscripts, strict=False)
    )
    einsum_string = f"{input_subscripts}->{output_subscript}"

    full_data = tuple(
        operators.raw_data.reshape(operators.basis[0].size, -1)
        for operators in op_as_tuple
    )
    data = cast("np.ndarray[Any, Any]", np.einsum(einsum_string, *full_data))  # type: ignore unknown
    full_coefficients = tuple(
        operators.basis[0].metadata().values[operators.basis[0].points]
        for operators in op_as_tuple
    )
    eigenvalues = outer_product(*full_coefficients)
    eigenvalue_basis = FundamentalBasis(EigenvalueMetadata(eigenvalues))

    return build_legacy_operator_list(
        tuple_basis(
            (eigenvalue_basis, recast_diagonal_basis(full_basis_1, full_basis_1))
        ),
        data,
    )


type NoiseOperatorList[
    M: BasisMetadata,
    B: Basis[Any, Any] = LegacyBasis[Metadata2D[M, M, None], np.complexfloating],
] = LegacyOperatorList[
    EigenvalueMetadata,
    M,
    np.complexfloating,
    LegacyTupleBasis2D[np.complexfloating, Basis[EigenvalueMetadata], B, None],
]

type DiagonalNoiseOperatorList[
    M: BasisMetadata,
    B: LegacyDiagonalBasis[
        np.complexfloating,
        LegacyBasis[BasisMetadata, np.complexfloating],
        LegacyBasis[BasisMetadata, np.complexfloating],
        None,
    ] = LegacyDiagonalBasis[
        np.complexfloating,
        LegacyBasis[M, np.complexfloating],
        LegacyBasis[M, np.complexfloating],
        None,
    ],
] = NoiseOperatorList[M, B]
