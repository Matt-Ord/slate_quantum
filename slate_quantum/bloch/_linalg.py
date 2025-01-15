from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from slate import BasisMetadata, FundamentalBasis, SimpleMetadata, StackedMetadata
from slate.basis import (
    BasisStateMetadata,
    BlockDiagonalBasis,
    DiagonalBasis,
    TupleBasis2D,
)

from slate_quantum import operator as _operator
from slate_quantum.bloch._transposed_basis import BlochTransposedBasis
from slate_quantum.metadata._repeat import RepeatedLengthMetadata
from slate_quantum.state._basis import EigenstateBasis

if TYPE_CHECKING:
    from slate_quantum.operator._operator import Operator

type BlochEigenstateBasis[M: RepeatedLengthMetadata, E] = EigenstateBasis[
    StackedMetadata[M, E],
    BlochTransposedBasis[np.complexfloating, M, E],
    BlockDiagonalBasis[
        np.generic,
        BasisMetadata,
        None,
        TupleBasis2D[
            np.generic,
            FundamentalBasis[SimpleMetadata],
            FundamentalBasis[
                BasisStateMetadata[BlochTransposedBasis[np.complexfloating, M, E]]
            ],
            None,
        ],
    ],
]


type DiagonalBlochBasis[M: RepeatedLengthMetadata, E] = DiagonalBasis[
    np.complexfloating,
    BlochEigenstateBasis[M, E],
    BlochEigenstateBasis[M, E],
    None,
]


def into_diagonal[M: RepeatedLengthMetadata, E](
    operator: Operator[
        StackedMetadata[M, E],
        np.complexfloating,
        BlockDiagonalBasis[
            np.complexfloating,
            M,
            E,
            TupleBasis2D[
                np.complexfloating,
                BlochTransposedBasis[np.complexfloating, M, E],
                BlochTransposedBasis[np.complexfloating, M, E],
                None,
            ],
        ],
    ],
) -> Operator[
    StackedMetadata[M, E],
    np.complexfloating,
    DiagonalBlochBasis[M, E],
]:
    diagonal = _operator.into_diagonal_hermitian(operator)
    return cast(
        "Operator[StackedMetadata[M, E], np.complexfloating, DiagonalBlochBasis[M, E]]",
        diagonal,
    )
