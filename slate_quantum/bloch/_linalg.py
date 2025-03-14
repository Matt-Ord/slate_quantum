from __future__ import annotations

from typing import TYPE_CHECKING, Never, cast

import numpy as np
from slate_core import (
    Array,
    Basis,
    Ctype,
    FundamentalBasis,
    SimpleMetadata,
    TupleBasis,
    TupleMetadata,
)
from slate_core.basis import (
    AsUpcast,
    BasisStateMetadata,
    BlockDiagonalBasis,
    DiagonalBasis,
)

from slate_quantum import operator as _operator
from slate_quantum.bloch._transposed_basis import BlochTransposedBasis
from slate_quantum.metadata._repeat import RepeatedLengthMetadata
from slate_quantum.operator._operator import OperatorMetadata
from slate_quantum.state._basis import EigenstateBasis

if TYPE_CHECKING:
    from slate_quantum.operator._operator import Operator

# TODO: name??
type BlochBasis[M: RepeatedLengthMetadata, E] = BlockDiagonalBasis[
    TupleBasis[
        tuple[
            FundamentalBasis[SimpleMetadata],
            FundamentalBasis[
                BasisStateMetadata[
                    BlochTransposedBasis[
                        TupleBasis[tuple[Basis[M], ...], E, Ctype[np.complexfloating]],
                        Ctype[np.complexfloating],
                    ]
                ]
            ],
        ],
        None,
        Ctype[np.generic],
    ],
    Ctype[np.generic],
]

type BlochEigenstateBasis[M: RepeatedLengthMetadata, E] = EigenstateBasis[
    Array[
        AsUpcast[
            BlochBasis[RepeatedLengthMetadata, E],
            TupleMetadata[tuple[M, ...], E],
            Ctype[np.complexfloating],
        ],
        np.dtype[np.complexfloating],
    ]
]


type DiagonalBlochBasis[
    M: RepeatedLengthMetadata,
    E,
    CT: Ctype[Never] = Ctype[Never],
] = AsUpcast[
    DiagonalBasis[
        TupleBasis[
            tuple[BlochEigenstateBasis[M, E], BlochEigenstateBasis[M, E]], None, CT
        ],
        CT,
    ],
    OperatorMetadata[TupleMetadata[tuple[M, ...], E]],
    CT,
]


def into_diagonal[M: RepeatedLengthMetadata, E](
    operator: Operator[
        AsUpcast[
            BlockDiagonalBasis[
                TupleBasis[
                    tuple[BlochTransposedBasis[M, E], BlochTransposedBasis[M, E]],
                    None,
                ],
            ],
            OperatorMetadata,
        ],
        np.dtype[np.complexfloating],
    ],
) -> Operator[
    DiagonalBlochBasis[M, E],
    np.dtype[np.complexfloating],
]:
    diagonal = _operator.into_diagonal_hermitian(operator)
    return cast(
        "Operator[TupleMetadata[tuple[M, ...], E], np.complexfloating, DiagonalBlochBasis[M, E]]",
        diagonal,
    )
