from typing import TYPE_CHECKING, cast

import numpy as np
from slate_core import (
    Array,
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
from slate_core.metadata import AxisDirections

from slate_quantum import operator as _operator
from slate_quantum.bloch._transposed_basis import (
    BlochOperatorBasis,
    BlochStateBasis,
)
from slate_quantum.metadata._repeat import RepeatedMetadata
from slate_quantum.state._basis import EigenstateBasis

if TYPE_CHECKING:
    from slate_quantum.operator._operator import (
        Operator,
        OperatorMetadata,
    )


type BlochEigenstateBasis[M: RepeatedMetadata, E: AxisDirections] = EigenstateBasis[
    Array[
        AsUpcast[
            BlockDiagonalBasis[
                TupleBasis[
                    tuple[
                        FundamentalBasis,
                        FundamentalBasis[BasisStateMetadata[BlochStateBasis[M, E]]],
                    ],
                    E,
                ]
            ],
            TupleMetadata[
                tuple[
                    SimpleMetadata,
                    BasisStateMetadata[BlochStateBasis[M, E]],
                ],
                None,
            ],
        ],
        np.dtype[np.complexfloating],
    ],
]


type DiagonalBlochBasis[M: RepeatedMetadata, E: AxisDirections] = DiagonalBasis[
    TupleBasis[
        tuple[
            BlochEigenstateBasis[M, E],
            BlochEigenstateBasis[M, E],
        ],
        None,
    ],
    Ctype[np.complexfloating],
]


def into_diagonal[M: RepeatedMetadata, E: AxisDirections](
    operator: Operator[BlochOperatorBasis[M, E], np.dtype[np.complexfloating]],
) -> Operator[
    AsUpcast[
        DiagonalBlochBasis[M, E], OperatorMetadata[TupleMetadata[tuple[M, ...], E]]
    ],
    np.dtype[np.complexfloating],
]:
    diagonal = _operator.into_diagonal_hermitian(operator)
    return cast(
        "Operator[AsUpcast[DiagonalBlochBasis[M, E], OperatorMetadata[TupleMetadata[tuple[M, ...], E]]], np.dtype[np.complexfloating]]",
        diagonal,
    )
