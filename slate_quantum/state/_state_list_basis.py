from __future__ import annotations

from slate_core import Basis, FundamentalBasis, TupleBasis
from slate_core.explicit_basis import ExplicitBasis
from slate_core.basis import RecastBasis
from slate_quantum.state._state import StateList


class StateListBasis[B:Basis](ExplicitBasis):
    """
    A class representing a state list basis.

    This represents a basis, where each 
    """

    def __init__(self, states: StateList[]) -> None:
        super().__init__(metadata, data)


type RecastStateListBasis[
   B:Basis,
] = RecastBasis[
    StateListBasis[BInner, BOuter, CT],
    BInner,
    BOuter,
    CT,
]

def as_state_basis(
    states: StateList[AsUpcast[TupleBasis[tuple[FundamentalBasis, B], None]]],
) -> RecastStateListBasis[B]:
    ...