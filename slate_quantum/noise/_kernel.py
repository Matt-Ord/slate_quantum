from __future__ import annotations

from typing import Any, Never, cast, override

import numpy as np
from slate_core import Ctype
from slate_core.basis import (
    AsUpcast,
    Basis,
    DiagonalBasis,
    FundamentalBasis,
    IsotropicBasis,
    RecastBasis,
    TupleBasis,
    TupleBasis2D,
    as_index_basis,
    as_tuple_basis,
)
from slate_core.metadata import (
    BasisMetadata,
)
from slate_core.util import slice_ignoring_axes

from slate_quantum._util import outer_product
from slate_quantum.metadata import EigenvalueMetadata
from slate_quantum.operator import (
    OperatorList,
    OperatorMetadata,
    SuperOperatorMetadata,
)
from slate_quantum.operator._diagonal import recast_diagonal_basis
from slate_quantum.operator._operator import (
    Operator,
    OperatorListBasis,
    SuperOperator,
    SuperOperatorBasis,
)

type NoiseKernel[B: Basis[SuperOperatorMetadata], DT: np.dtype[np.generic]] = (
    SuperOperator[B, DT]
)
type NoiseKernelWithMetadata[M: BasisMetadata, DT: np.dtype[np.generic]] = NoiseKernel[
    SuperOperatorBasis[M], DT
]


def from_noise_operators[M: BasisMetadata, DT: np.dtype[np.generic]](
    operators: OperatorList[
        OperatorListBasis[EigenvalueMetadata, OperatorMetadata[M]], DT
    ],
) -> NoiseKernel[SuperOperatorBasis[M], DT]:
    converted = operators.with_basis(
        as_tuple_basis(operators.basis).upcast()
    ).assert_ok()
    converted_inner = (
        converted.with_operator_basis(
            as_tuple_basis(converted.basis.inner.children[1]).upcast()
        )
        .assert_ok()
        .with_list_basis(as_index_basis(converted.basis.inner.children[0]))
        .assert_ok()
    )
    operators_data = converted_inner.raw_data.reshape(
        converted.basis.inner.children[0].size,
        *converted_inner.basis.inner.children[1].shape,
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
        TupleBasis((converted_inner.basis[1], converted_inner.basis[1])), data
    ).ok()


def diagonal_kernel_basis() -> None: ...


type DiagonalKernelBasisInner[
    B0: Basis = Basis,
    B1: Basis = Basis,
    CT: Ctype[Never] = Ctype[Never],
] = TupleBasis[
    tuple[
        DiagonalBasis[TupleBasis[tuple[B0, B0], None, CT]],
        DiagonalBasis[TupleBasis[tuple[B1, B1], None, CT]],
    ],
    None,
    CT,
]

type DiagonalKernelBasis[
    B0: Basis = Basis,
    B1: Basis = Basis,
    BOuter: Basis = Basis,
    CT: Ctype[Never] = Ctype[Never],
] = AsUpcast[
    RecastBasis[
        DiagonalKernelBasisInner[B0, B1, CT],
        TupleBasis[tuple[B0, B0], None, CT],
        BOuter,
        CT,
    ],
    SuperOperatorMetadata,
    CT,
]

type DiagonalKernelBasisWithMetadata[
    M: BasisMetadata,
    CT: Ctype[Never] = Ctype[Never],
] = DiagonalKernelBasis[
    Basis[M],
    Basis[M],
    SuperOperatorBasis[M],
    CT,
]

type DiagonalNoiseKernel[B: DiagonalKernelBasis, DT: np.dtype[np.generic]] = (
    NoiseKernel[B, DT]
)

type DiagonalKernelWithMetadata[
    M: BasisMetadata,
    CT: Ctype[Never],
    DT: np.dtype[np.generic],
] = DiagonalNoiseKernel[
    DiagonalKernelBasisWithMetadata[M, CT],
    DT,
]


def from_diagonal_operators[M_: BasisMetadata, DT_: np.generic](
    operators: OperatorList[EigenvalueMetadata, M_, DT_],
) -> DiagonalKernelWithMetadata[M_, Ctype[np.generic], DT_]:
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
    return DiagonalNoiseKernel[M_, DT_](converted_inner.basis[1].inner, data)


def with_outer_basis[M_: BasisMetadata, DT_: np.generic](
    self: DiagonalNoiseKernel[M_, DT_],
    basis: Basis[Metadata2D[M_, M_, Any], Any],
) -> DiagonalNoiseKernel[M_, DT_]:
    """Get the Potential with the outer recast basis set to basis."""
    return DiagonalNoiseKernel[M_, DT_](
        basis,
        self.basis.outer_recast.__convert_vector_into__(self.raw_data, basis),
    )


type IsotropicKernelBasis[
    M: BasisMetadata,
    DT: np.generic,
    OuterB: Basis[BasisMetadata, Any] = Basis[M, DT],
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
type IsotropicKernelBasis[
    B0: Basis = Basis,
    B1: Basis = Basis,
    BOuter: Basis = Basis,
    CT: Ctype[Never] = Ctype[Never],
] = AsUpcast[
    RecastBasis[
        IsotropicKernelBasisInner[B0, B1, CT],
        IsotropicBasis[TupleBasis[tuple[B0, B0], None, CT]],
        BOuter,
        CT,
    ],
    SuperOperatorMetadata,
    CT,
]

type IsotropicKernelBasisWithMetadata[
    M: BasisMetadata,
    CT: Ctype[Never] = Ctype[Never],
] = DiagonalKernelBasis[
    Basis[M],
    Basis[M],
    SuperOperatorBasis[M],
    CT,
]

type IsotropicNoiseKernel[B: IsotropicKernelBasis, DT: np.dtype[np.generic]] = (
    NoiseKernel[B, DT]
)

type IsotropicKernelWithMetadata[
    M: BasisMetadata,
    CT: Ctype[Never],
    DT: np.dtype[np.generic],
] = IsotropicNoiseKernel[
    IsotropicKernelBasisWithMetadata[M, CT],
    DT,
]


def isotropic_kernel_basis() -> None: ...


class IsotropicNoiseKernel[
    M: BasisMetadata,
    DT: np.generic,
    B: Basis[SuperOperatorMetadata, Any] = IsotropicKernelBasis[M, DT],
](DiagonalNoiseKernel[M, DT, B]):
    r"""
    Represents a noise kernel which is isotropic and diagonal.

    In this case, the correlation between any pair of states depends only on
    the difference between the two states. We therefore store the kernel
    relating to only a single state.

    The full kernel has the basis ((a, a), (a, a)), and the isotropic kernel
    is indexed according to the basis ((a, a), (0, 0)).
    """

    def __init__[
        M_: BasisMetadata,
        DT_: np.generic,
    ](
        self: IsotropicNoiseKernel[M_, DT_, IsotropicKernelBasis[M_, DT_]],
        basis: Basis[M_, DT_],
        data: np.ndarray[Any, np.dtype[DT]],
    ) -> None:
        outer_recast = basis
        inner_recast = basis
        inner = isotropic_basis((basis, basis.dual_basis()))

        recast = RecastBasis(inner, inner_recast, outer_recast)
        super().__init__(cast("Any", recast), cast("Any", data))

    def with_isotropic_basis[
        M_: BasisMetadata,
        DT_: np.generic,
    ](
        self: IsotropicNoiseKernel[
            M_,
            DT_,
        ],
        basis: Basis[M_, Any],
    ) -> IsotropicNoiseKernel[M_, DT_]:
        """Get the Potential with the outer recast basis set to basis."""
        return IsotropicNoiseKernel(
            basis,
            self.basis.outer_recast.outer_recast.__convert_vector_into__(
                self.raw_data, basis
            ),
        )

    @staticmethod
    def from_diagonal_kernel[M_: BasisMetadata, DT_: np.generic](
        kernel: DiagonalNoiseKernel[M_, DT_],
    ) -> IsotropicNoiseKernel[M_, DT_]:
        """Build a diagonal kernel from operators."""
        basis = as_tuple_basis(kernel.basis.outer_recast)[0]
        converted = kernel.with_outer_basis(
            isotropic_basis((basis, basis.dual_basis()))
        )
        return IsotropicNoiseKernel(basis, converted.raw_data)

    @staticmethod
    @override
    def from_operators[M_: BasisMetadata, DT_: np.generic](
        operators: OperatorList[EigenvalueMetadata, M_, DT_],
    ) -> IsotropicNoiseKernel[M_, DT_]:
        """Build a diagonal kernel from operators."""
        diagonal_kernel = DiagonalNoiseKernel.from_operators(operators)
        return IsotropicNoiseKernel.from_diagonal_kernel(diagonal_kernel)

    def unwrap[
        M_: BasisMetadata,
        DT_: np.generic,
    ](
        self: IsotropicNoiseKernel[
            M_,
            DT_,
        ],
    ) -> DiagonalNoiseKernel[M_, DT_]:
        """Unwrap the isotropic kernel into a diagonal kernel."""
        return DiagonalNoiseKernel[M_, DT_, Any](
            cast("Any", self.basis.outer_recast.inner),
            self.raw_data,
        )


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

    return IsotropicNoiseKernel(
        TupleBasis(full_basis, None), outer_product(*full_data).ravel()
    )


def as_axis_kernel_from_isotropic[
    M: BasisMetadata,
    DT: np.complexfloating,
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
            np.complexfloating,
        ],
        ...,
    ],
    extra: E,
) -> OperatorList[
    EigenvalueMetadata,
    TupleMetadata[tuple[M, ...], E],
    np.complex128,
    TupleBasis2D[
        np.complexfloating,
        FundamentalBasis[EigenvalueMetadata],
        RecastDiagonalOperatorBasis[
            TupleMetadata[tuple[M, ...], E],
            np.complexfloating,
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

    full_basis_1 = TupleBasis(
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

    return OperatorList(
        TupleBasis(
            (eigenvalue_basis, recast_diagonal_basis(full_basis_1, full_basis_1))
        ),
        data,
    )


type NoiseOperatorList[
    M: BasisMetadata,
    B: Basis[
        Metadata2D[BasisMetadata, BasisMetadata, None], np.complexfloating
    ] = Basis[Metadata2D[M, M, None], np.complexfloating],
] = OperatorList[
    EigenvalueMetadata,
    M,
    np.complexfloating,
    TupleBasis2D[np.complexfloating, Basis[EigenvalueMetadata, Any], B, None],
]

type DiagonalNoiseOperatorList[
    M: BasisMetadata,
    B: DiagonalBasis[
        np.complexfloating,
        Basis[BasisMetadata, np.complexfloating],
        Basis[BasisMetadata, np.complexfloating],
        None,
    ] = DiagonalBasis[
        np.complexfloating,
        Basis[M, np.complexfloating],
        Basis[M, np.complexfloating],
        None,
    ],
] = NoiseOperatorList[M, B]
