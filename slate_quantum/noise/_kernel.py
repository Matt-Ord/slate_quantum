from __future__ import annotations

from typing import Any, cast, override

import numpy as np
from slate.basis import (
    Basis,
    DiagonalBasis,
    FundamentalBasis,
    IsotropicBasis,
    RecastBasis,
    TupleBasis2D,
    as_index_basis,
    as_tuple_basis,
    diagonal_basis,
    isotropic_basis,
    tuple_basis,
)
from slate.metadata import (
    BasisMetadata,
    Metadata2D,
    StackedMetadata,
)
from slate.util import slice_ignoring_axes

from slate_quantum._util import outer_product
from slate_quantum.model import EigenvalueMetadata
from slate_quantum.model.operator import (
    OperatorList,
    OperatorMetadata,
    RecastDiagonalOperatorBasis,
    SuperOperator,
    SuperOperatorMetadata,
)
from slate_quantum.model.operator._diagonal import recast_diagonal_basis


class NoiseKernel[
    M: BasisMetadata,
    DT: np.generic,
    B: Basis[SuperOperatorMetadata, Any] = Basis[SuperOperatorMetadata[M], DT],
](SuperOperator[M, DT, B]):
    r"""Represents a noise kernel which is diagonal."""

    @staticmethod
    def from_operators[_M: BasisMetadata, _DT: np.generic](
        operators: OperatorList[EigenvalueMetadata, _M, _DT],
    ) -> NoiseKernel[_M, _DT]:
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
            operators_data.astype(np.complex128),
        )
        return NoiseKernel(
            tuple_basis((converted_inner.basis[1], converted_inner.basis[1])), data
        )


type DiagonalKernelBasis[
    M: BasisMetadata,
    DT: np.generic,
    OuterB: Basis[OperatorMetadata, Any] = Basis[OperatorMetadata[M], DT],  # noqa: E251
] = RecastBasis[
    SuperOperatorMetadata[M],
    Metadata2D[M, M, None],
    DT,
    TupleBasis2D[
        np.generic,
        DiagonalBasis[Any, Basis[M, Any], Basis[M, Any], None],
        DiagonalBasis[Any, Basis[M, Any], Basis[M, Any], None],
        None,
    ],
    OuterB,
]


class DiagonalNoiseKernel[
    M: BasisMetadata,
    DT: np.generic,
    B: Basis[SuperOperatorMetadata, Any] = DiagonalKernelBasis[M, DT],
](NoiseKernel[M, DT, B]):
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
        _M: BasisMetadata,
        _DT: np.generic,
        _B: Basis[SuperOperatorMetadata, Any] = DiagonalKernelBasis[_M, _DT],
    ](
        self: DiagonalNoiseKernel[_M, _DT, _B],
        basis: Basis[Metadata2D[_M, _M, None], Any],
        data: np.ndarray[Any, np.dtype[DT]],
    ) -> None:
        outer_recast = basis
        inner_recast = as_tuple_basis(basis)
        b0 = inner_recast.children[0]
        b1 = inner_recast.children[1]
        inner = tuple_basis((diagonal_basis((b0, b0)), diagonal_basis((b1, b1))))
        actual_basis = RecastBasis(inner, inner_recast, outer_recast)
        super().__init__(cast("Any", actual_basis), cast("Any", data))

    def with_outer_basis[_M: BasisMetadata, _DT: np.generic](
        self: DiagonalNoiseKernel[_M, _DT],
        basis: Basis[Metadata2D[_M, _M, Any], Any],
    ) -> DiagonalNoiseKernel[_M, _DT]:
        """Get the Potential with the outer recast basis set to basis."""
        return DiagonalNoiseKernel[_M, _DT](
            basis,
            self.basis.outer_recast.__convert_vector_into__(self.raw_data, basis),
        )

    @staticmethod
    @override
    def from_operators[_M: BasisMetadata, _DT: np.generic](
        operators: OperatorList[EigenvalueMetadata, _M, _DT],
    ) -> DiagonalNoiseKernel[_M, _DT]:
        """Build a diagonal kernel from operators."""
        converted = operators.with_basis(as_tuple_basis(operators.basis))
        converted_inner = converted.with_operator_basis(
            DiagonalBasis(as_tuple_basis(converted.basis[1]))
        ).with_list_basis(as_index_basis(converted.basis[0]))

        operators_data = converted_inner.raw_data.reshape(converted.basis[0].size, -1)
        data = cast(
            "Any",
            np.einsum(  # type:ignore  unknown
                "a,ai,aj->ij",
                converted.basis[0].metadata().values[converted_inner.basis[0].points],
                np.conj(operators_data),
                operators_data,  # type:ignore DT not numeric
            ),
        )
        return DiagonalNoiseKernel[_M, _DT](converted_inner.basis[1].inner, data)


type IsotropicKernelBasis[
    M: BasisMetadata,
    DT: np.generic,
    OuterB: Basis[BasisMetadata, Any] = Basis[M, DT],  # noqa: E251
] = DiagonalKernelBasis[
    M,
    DT,
    RecastBasis[
        OperatorMetadata[M],
        M,
        DT,
        IsotropicBasis[Any, Basis[M, Any], Basis[M, Any], None],
        OuterB,
    ],
]


class IsotropicNoiseKernel[
    M: BasisMetadata,
    DT: np.generic,
    B: Basis[SuperOperatorMetadata, Any] = IsotropicKernelBasis[M, DT],
](DiagonalNoiseKernel[M, DT, B]):
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
        self: IsotropicNoiseKernel[_M, _DT, IsotropicKernelBasis[_M, _DT]],
        basis: Basis[_M, _DT],
        data: np.ndarray[Any, np.dtype[DT]],
    ) -> None:
        outer_recast = basis
        inner_recast = basis
        inner = isotropic_basis((basis, basis.dual_basis()))

        recast = RecastBasis(inner, inner_recast, outer_recast)
        super().__init__(cast("Any", recast), cast("Any", data))

    def with_isotropic_basis[
        _M: BasisMetadata,
        _DT: np.generic,
    ](
        self: IsotropicNoiseKernel[
            _M,
            _DT,
        ],
        basis: Basis[_M, Any],
    ) -> IsotropicNoiseKernel[_M, _DT]:
        """Get the Potential with the outer recast basis set to basis."""
        return IsotropicNoiseKernel(
            basis,
            self.basis.outer_recast.outer_recast.__convert_vector_into__(
                self.raw_data, basis
            ),
        )

    @staticmethod
    def from_diagonal_kernel[_M: BasisMetadata, _DT: np.generic](
        kernel: DiagonalNoiseKernel[_M, _DT],
    ) -> IsotropicNoiseKernel[_M, _DT]:
        """Build a diagonal kernel from operators."""
        basis = as_tuple_basis(kernel.basis.outer_recast)[0]
        converted = kernel.with_outer_basis(
            isotropic_basis((basis, basis.dual_basis()))
        )
        return IsotropicNoiseKernel(basis, converted.raw_data)

    @staticmethod
    @override
    def from_operators[_M: BasisMetadata, _DT: np.generic](
        operators: OperatorList[EigenvalueMetadata, _M, _DT],
    ) -> IsotropicNoiseKernel[_M, _DT]:
        """Build a diagonal kernel from operators."""
        diagonal_kernel = DiagonalNoiseKernel.from_operators(operators)
        return IsotropicNoiseKernel.from_diagonal_kernel(diagonal_kernel)

    def unwrap[
        _M: BasisMetadata,
        _DT: np.generic,
    ](
        self: IsotropicNoiseKernel[
            _M,
            _DT,
        ],
    ) -> DiagonalNoiseKernel[_M, _DT]:
        """Unwrap the isotropic kernel into a diagonal kernel."""
        return DiagonalNoiseKernel[_M, _DT, Any](
            cast("Any", self.basis.outer_recast.inner),
            self.raw_data,
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
    outer_as_tuple = as_tuple_basis(kernel.basis.outer_recast.outer_recast)
    converted = kernel.with_isotropic_basis(outer_as_tuple)
    n_axis = len(outer_as_tuple.shape)

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


def get_diagonal_noise_operators_from_axis[M: BasisMetadata, E](
    operators_list: tuple[
        OperatorList[
            EigenvalueMetadata,
            M,
            np.complex128,
        ],
        ...,
    ],
    extra: E,
) -> OperatorList[
    EigenvalueMetadata,
    StackedMetadata[M, E],
    np.complex128,
    TupleBasis2D[
        np.complex128,
        FundamentalBasis[EigenvalueMetadata],
        RecastDiagonalOperatorBasis[
            StackedMetadata[M, E],
            np.complex128,
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

    full_basis_1 = tuple_basis(
        tuple(operators.basis[1].inner[1] for operators in op_as_tuple_nested), extra
    )

    # for example, in 2d this is ij,kl ->  # cSpell:ignore
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
    data = cast("np.ndarray[Any, Any]", np.einsum(einsum_string, *full_data))  # type: ignore unknown
    full_coefficients = tuple(
        operators.basis[0].metadata().values[operators.basis[0].points]
        for operators in op_as_tuple
    )
    eigenvalues = outer_product(*full_coefficients)
    eigenvalue_basis = FundamentalBasis(EigenvalueMetadata(eigenvalues))

    return OperatorList(
        tuple_basis(
            (eigenvalue_basis, recast_diagonal_basis(full_basis_1, full_basis_1))
        ),
        data,
    )


type NoiseOperatorList[
    M: Metadata2D[BasisMetadata, BasisMetadata, None],
    B: Basis[Metadata2D[BasisMetadata, BasisMetadata, None], np.complex128] = Basis[  # noqa: E251
        Metadata2D[BasisMetadata, BasisMetadata, None], np.complex128
    ],
] = OperatorList[
    EigenvalueMetadata,
    M,
    np.complex128,
    TupleBasis2D[np.complex128, FundamentalBasis[EigenvalueMetadata], B, None],
]

type DiagonalNoiseOperatorList[
    M: Metadata2D[BasisMetadata, BasisMetadata, None],
    B: DiagonalBasis[
        np.complex128,
        Basis[BasisMetadata, np.complex128],
        Basis[BasisMetadata, np.complex128],
        None,
    ] = DiagonalBasis[  # noqa: E251
        np.complex128,
        Basis[M, np.complex128],
        Basis[M, np.complex128],
        None,
    ],
] = NoiseOperatorList[M, B]
