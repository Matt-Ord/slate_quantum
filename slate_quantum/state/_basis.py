from __future__ import annotations

from typing import override

import numpy as np
from slate.explicit_basis import ExplicitUnitaryBasis
from slate.metadata import BasisMetadata

from slate_quantum.state._state import StateList


class EigenstateBasis[M: BasisMetadata](ExplicitUnitaryBasis[M, np.complex128]):
    """A basis with data stored as eigenstates."""

    @property
    @override
    def eigenvectors(
        self,
    ) -> StateList[BasisMetadata, M]:
        """Get the eigenstates of the basis."""
        states = super().eigenvectors
        return StateList(states.basis, states.raw_data)
