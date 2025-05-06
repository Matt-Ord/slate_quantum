from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from slate_core import BasisMetadata, FundamentalBasis, SimpleMetadata
from slate_core.basis import (
    BasisStateMetadata,
)

from slate_quantum import operator as _operator
from slate_quantum._util.legacy import (
    LegacyBlockDiagonalBasis,
    LegacyDiagonalBasis,
    LegacyTupleBasis2D,
    StackedMetadata,
)
from slate_quantum.bloch._transposed_basis import LegacyBlochTransposedBasis
from slate_quantum.metadata._repeat import RepeatedLengthMetadata
from slate_quantum.state._basis import LegacyEigenstateBasis

if TYPE_CHECKING:
    from slate_quantum.operator._operator import LegacyOperator

type BlochEigenstateBasis[M: RepeatedLengthMetadata, E] = LegacyEigenstateBasis[
    StackedMetadata[M, E],
    LegacyBlochTransposedBasis[np.complexfloating, M, E],
    LegacyBlockDiagonalBasis[
        np.generic,
        BasisMetadata,
        None,
        LegacyTupleBasis2D[
            np.generic,
            FundamentalBasis[SimpleMetadata],
            FundamentalBasis[
                BasisStateMetadata[LegacyBlochTransposedBasis[np.complexfloating, M, E]]
            ],
            None,
        ],
    ],
]


type DiagonalBlochBasis[M: RepeatedLengthMetadata, E] = LegacyDiagonalBasis[
    np.complexfloating,
    BlochEigenstateBasis[M, E],
    BlochEigenstateBasis[M, E],
    None,
]


def into_diagonal[M: RepeatedLengthMetadata, E](
    operator: LegacyOperator[
        StackedMetadata[M, E],
        np.complexfloating,
        LegacyBlockDiagonalBasis[
            np.complexfloating,
            M,
            E,
            LegacyTupleBasis2D[
                np.complexfloating,
                LegacyBlochTransposedBasis[np.complexfloating, M, E],
                LegacyBlochTransposedBasis[np.complexfloating, M, E],
                None,
            ],
        ],
    ],
) -> LegacyOperator[
    StackedMetadata[M, E],
    np.complexfloating,
    DiagonalBlochBasis[M, E],
]:
    diagonal = _operator.into_diagonal_hermitian(operator)
    return cast(
        "LegacyOperator[StackedMetadata[M, E], np.complexfloating, DiagonalBlochBasis[M, E]]",
        diagonal,
    )
