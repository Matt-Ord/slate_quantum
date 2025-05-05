from __future__ import annotations

from typing import Any

import numpy as np
from slate_core.basis import (
    Basis,
)
from slate_core.metadata import BasisMetadata

from slate_quantum.operator._operator import (
    OperatorMetadata,
    SuperOperator,
    SuperOperatorBasis,
)

type SuperOperatorMetadata[M: BasisMetadata = BasisMetadata] = OperatorMetadata[
    OperatorMetadata[M],
]


type LegacySuperOperator[
    M: BasisMetadata,
    DT: np.generic,
    B: Basis[
        SuperOperatorMetadata,
        Any,
    ] = Basis[SuperOperatorMetadata[M], Any],
] = SuperOperator[
    SuperOperatorBasis[M],
    np.dtype[DT],
]
