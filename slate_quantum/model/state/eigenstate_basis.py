from __future__ import annotations

from typing import TYPE_CHECKING, override

import numpy as np
from slate.explicit_basis import ExplicitUnitaryBasis
from slate.metadata import BasisMetadata

from ._state import StateList

if TYPE_CHECKING:
    from slate.basis import Basis
    from slate.basis.stacked import VariadicTupleBasis


class EigenstateBasis[M: BasisMetadata](
    ExplicitUnitaryBasis[BasisMetadata, np.complex128]
):
    """A basis with data stored as eigenstates."""

    @property
    def states(
        self,
    ) -> StateList[
        BasisMetadata,
        VariadicTupleBasis[
            np.complex128,
            Basis[BasisMetadata, np.generic],
            Basis[M, np.complex128],
            None,
        ],
    ]:
        """Get the eigenstates of the basis."""
        return StateList(self._data.basis, self._data.raw_data)

    @override
    def conjugate_basis(self) -> EigenstateBasis[M]:
        return EigenstateBasis(
            self._data,
            direction="forward" if self.direction == "backward" else "backward",
            data_id=self._data_id,
        )
