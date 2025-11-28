from typing import TYPE_CHECKING

import numpy as np
from slate_core import (
    SimpleMetadata,
    TupleMetadata,
)
from slate_core.basis import AsUpcast, TupleBasis
from slate_core.metadata import BasisMetadata

from slate_quantum.state import StateList

if TYPE_CHECKING:
    from slate_core.basis import Basis

    from slate_quantum.metadata import TimeMetadata

type RealizationMetadata[MT: TimeMetadata, M: BasisMetadata] = TupleMetadata[
    tuple[MT, M], None
]
type RealizationBasis[MT: TimeMetadata, M: BasisMetadata] = Basis[
    RealizationMetadata[MT, M]
]


type RealizationListIndexMetadata[MT: TimeMetadata] = TupleMetadata[
    tuple[SimpleMetadata, MT], None
]
type RealizationListMetadata[MT: TimeMetadata, M: BasisMetadata] = TupleMetadata[
    tuple[RealizationListIndexMetadata[MT], M], None
]
type RealizationListBasis[MT: TimeMetadata, M: BasisMetadata] = AsUpcast[
    TupleBasis[tuple[Basis[RealizationListIndexMetadata[MT]], Basis[M]], None],
    TupleMetadata[tuple[RealizationListIndexMetadata[MT], M], None],
]
type RealizationList[
    B: RealizationListBasis[TimeMetadata, BasisMetadata],
    DT: np.dtype[np.generic] = np.dtype[np.complexfloating],
] = StateList[B, DT]


def select_realization[MT: TimeMetadata, M: BasisMetadata, DT: np.dtype[np.generic]](
    states: StateList[Basis[RealizationListMetadata[MT, M]], DT],
    idx: int = 0,
) -> StateList[RealizationBasis[MT, M], DT]:
    """Select a realization from a state list."""
    return states[(idx, slice(None)), :]
