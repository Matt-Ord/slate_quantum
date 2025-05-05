from __future__ import annotations

from typing import Any, Never, cast

import numpy as np
from slate_core import Ctype, TupleMetadata, basis
from slate_core.basis import (
    AsUpcast,
    Basis,
    DiagonalBasis,
    FundamentalBasis,
    IsotropicBasis,
    RecastBasis,
    TupleBasis,
)
from slate_core.metadata import (
    BasisMetadata,
)

from slate_quantum.metadata import EigenvalueMetadata
from slate_quantum.operator import (
    OperatorList,
    OperatorMetadata,
    SuperOperatorMetadata,
)
from slate_quantum.operator._diagonal import (
    recast_diagonal_basis_with_metadata,
)
from slate_quantum.operator._operator import (
    OperatorBasis,
    OperatorConversion,
    OperatorListBasis,
    SuperOperator,
    SuperOperatorBasis,
)
from slate_quantum.util._prod import outer_product

type NoiseKernel[B: Basis[SuperOperatorMetadata], DT: np.dtype[np.generic]] = (
    SuperOperator[B, DT]
)
type NoiseKernelWithMetadata[M: BasisMetadata, DT: np.dtype[np.generic]] = NoiseKernel[
    SuperOperatorBasis[M], DT
]


def diagonal_kernel_basis[M: BasisMetadata, CT: Ctype[Never]](
    outer_recast: OperatorBasis[M, CT],
) -> DiagonalKernelBasisWithMetadata[M, Ctype[Never]]:
    inner_recast = basis.as_tuple(outer_recast)
    b0 = inner_recast.children[0]
    b1 = inner_recast.children[1]
    inner = TupleBasis(
        (DiagonalBasis(TupleBasis((b0, b0))), DiagonalBasis(TupleBasis((b1, b1))))
    )

    meta = TupleMetadata(
        (inner.children[0].metadata(), inner.children[1].metadata()), None
    )
    return AsUpcast(RecastBasis(inner, inner_recast, outer_recast), meta)


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
    M: SuperOperatorMetadata = SuperOperatorMetadata,
] = AsUpcast[
    RecastBasis[
        DiagonalKernelBasisInner[B0, B1, CT],
        TupleBasis[tuple[B0, B0], None, CT],
        BOuter,
        CT,
    ],
    M,
    CT,
]

type DiagonalKernelBasisWithMetadata[
    M: BasisMetadata,
    CT: Ctype[Never] = Ctype[Never],
] = DiagonalKernelBasis[
    Basis[M], Basis[M], OperatorBasis[M], CT, SuperOperatorMetadata[M]
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


def with_outer_basis[M1: BasisMetadata, M: BasisMetadata, DT: np.dtype[np.generic]](
    kernel: DiagonalKernelWithMetadata[M1, Ctype[Never], DT],
    basis: OperatorBasis[M],
) -> OperatorConversion[
    SuperOperatorMetadata[M1], DiagonalKernelBasisWithMetadata[M], DT
]:
    """Get the Potential with the outer recast basis set to basis."""
    final_basis = diagonal_kernel_basis(basis)
    return kernel.with_basis(final_basis)


type IsotropicKernelBasisOuter[B: Basis = Basis] = RecastBasis[
    IsotropicBasis[TupleBasis[tuple[B, B], None]], B, B
]
type IsotropicKernelBasis[
    B0: Basis = Basis,
    B1: Basis = Basis,
    BOuter: IsotropicKernelBasisOuter = IsotropicKernelBasisOuter,
    CT: Ctype[Never] = Ctype[Never],
    M: SuperOperatorMetadata = SuperOperatorMetadata,
] = DiagonalKernelBasis[B0, B1, BOuter, CT, M]

type IsotropicKernelBasisWithMetadata[
    M: BasisMetadata,
    CT: Ctype[Never] = Ctype[Never],
] = IsotropicKernelBasis[
    Basis[M],
    Basis[M],
    IsotropicKernelBasisOuter[Basis[M]],
    CT,
    SuperOperatorMetadata[M],
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


def isotropic_kernel_basis[M: BasisMetadata, CT: Ctype[Never]](
    outer_basis: Basis[M, CT],
) -> IsotropicKernelBasisWithMetadata[M, CT]:
    outer_recast = outer_basis
    inner_recast = outer_basis
    inner = IsotropicBasis(TupleBasis((outer_basis, outer_basis.dual_basis())))
    recast: IsotropicKernelBasisOuter[Basis[M]] = RecastBasis(
        inner, inner_recast, outer_recast
    )
    # recast does not have the correct M type here...
    return diagonal_kernel_basis(recast)


def with_isotropic_basis[M: BasisMetadata, CT: Ctype[Never], DT: np.dtype[np.generic]](
    kernel: IsotropicKernelWithMetadata[M, CT, DT],
    outer_basis: Basis[M, CT],
) -> IsotropicKernelWithMetadata[M, CT, DT]:
    basis = isotropic_kernel_basis(outer_basis)
    return kernel.with_basis(basis).assert_ok()


type AxisKernel[
    M: BasisMetadata,
    CT: Ctype[Never],
    DT: np.dtype[np.complexfloating],
] = tuple[
    IsotropicKernelWithMetadata[M, CT, DT],
    ...,
]


def get_diagonal_noise_operators_from_axis[M: BasisMetadata, E](
    operators_list: tuple[
        OperatorList[
            OperatorListBasis[EigenvalueMetadata, OperatorMetadata[M]],
            np.dtype[np.complexfloating],
        ],
        ...,
    ],
    extra: E,
) -> OperatorList[
    OperatorListBasis[
        EigenvalueMetadata, OperatorMetadata[TupleMetadata[tuple[M, ...], E]]
    ],
    np.dtype[np.complexfloating],
]:
    """Convert axis operators into full operators."""
    op_as_tuple = tuple(
        operators.with_basis(basis.as_tuple(operators.basis).upcast()).assert_ok()
        for operators in operators_list
    )
    op_as_tuple_nested = tuple(
        operators.with_list_basis(basis.as_index(operators.basis.inner.children[0]))
        .assert_ok()
        .with_operator_basis(
            DiagonalBasis(basis.as_tuple(operators.basis.inner.children[1])).upcast()
        )
        .assert_ok()
        for operators in op_as_tuple
    )

    full_basis_1 = TupleBasis(
        tuple(
            operators.basis.inner.children[1].inner.inner.children[1]
            for operators in op_as_tuple_nested
        ),
        extra,
    ).upcast()

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
        operators.raw_data.reshape(operators.basis.inner.children[0].size, -1)
        for operators in op_as_tuple
    )
    data = cast("np.ndarray[Any, Any]", np.einsum(einsum_string, *full_data))  # type: ignore unknown
    full_coefficients = tuple(
        operators.basis.metadata()
        .children[0]
        .values[operators.basis.inner.children[0].points]
        for operators in op_as_tuple
    )
    eigenvalues = outer_product(*full_coefficients)
    eigenvalue_basis = FundamentalBasis(EigenvalueMetadata(eigenvalues))

    return OperatorList.build(
        TupleBasis(
            (
                eigenvalue_basis,
                recast_diagonal_basis_with_metadata(full_basis_1, full_basis_1),
            )
        ).upcast(),
        data,
    ).assert_ok()


type NoiseOperatorList[
    M: BasisMetadata,
    DT: np.dtype[np.complexfloating] = np.dtype[np.complexfloating],
] = OperatorList[OperatorListBasis[EigenvalueMetadata, OperatorMetadata[M]], DT]
