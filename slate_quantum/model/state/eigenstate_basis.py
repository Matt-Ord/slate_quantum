from __future__ import annotations

from typing import TYPE_CHECKING, override

import numpy as np
from slate.explicit_basis import ExplicitUnitaryBasis
from slate.metadata import BasisMetadata, Metadata2D

from ._state import StateList

if TYPE_CHECKING:
    from slate.basis import Basis, TupleBasis2D


class EigenstateBasis[M: BasisMetadata](ExplicitUnitaryBasis[M, np.complex128]):
    """A basis with data stored as eigenstates."""

    @property
    @override
    def states(
        self,
    ) -> StateList[
        Metadata2D[BasisMetadata, M, None],
        TupleBasis2D[
            np.complex128,
            Basis[BasisMetadata, np.generic],
            Basis[M, np.complex128],
            None,
        ],
    ]:
        """Get the eigenstates of the basis."""
        states = super().states
        return StateList(states.basis, states.raw_data)

    @override
    def conjugate_basis(self) -> EigenstateBasis[M]:
        return EigenstateBasis(
            self.states,
            direction="forward" if self.direction == "backward" else "backward",
            data_id=self._data_id,
        )
