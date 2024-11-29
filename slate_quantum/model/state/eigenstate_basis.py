from __future__ import annotations

from typing import TYPE_CHECKING, override

import numpy as np
from slate.explicit_basis import ExplicitUnitaryBasis
from slate.metadata import BasisMetadata

from slate_quantum.model.state._state import StateList

if TYPE_CHECKING:
    from slate.basis import Basis, TupleBasis2D


class EigenstateBasis[M: BasisMetadata](ExplicitUnitaryBasis[M, np.complex128]):
    """A basis with data stored as eigenstates."""

    @property
    @override
    def eigenvectors(
        self,
    ) -> StateList[
        BasisMetadata,
        M,
        TupleBasis2D[
            np.complex128,
            Basis[BasisMetadata, np.generic],
            Basis[M, np.complex128],
            None,
        ],
    ]:
        """Get the eigenstates of the basis."""
        states = super().eigenvectors
        return StateList(states.basis, states.raw_data)
