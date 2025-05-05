from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast, override

import numpy as np
from slate_core import Basis, SimpleMetadata
from slate_core.explicit_basis import ExplicitUnitaryBasis
from slate_core.metadata import BasisMetadata

from slate_quantum._util.legacy import LegacyArray, LegacyBasis, Metadata2D
from slate_quantum.state._state import build_legacy_state_list

if TYPE_CHECKING:
    import uuid

    from slate_core.basis import BasisStateMetadata

    from slate_quantum.state._state import LegacyStateList

type Direction = Literal["forward", "backward"]


class EigenstateBasis[
    M: BasisMetadata,
    B: LegacyBasis[Any, np.complexfloating] = LegacyBasis[M, np.complexfloating],
    BTransform: Basis[Any, Any] = Basis[
        Metadata2D[SimpleMetadata, BasisStateMetadata[B], None], Any
    ],
](ExplicitUnitaryBasis[Any, Any]):
    """A basis with data stored as eigenstates."""

    def __init__[B1: Basis[Any, Any]](
        self: EigenstateBasis[Any, B1],
        matrix: LegacyArray[
            Metadata2D[BasisMetadata, BasisStateMetadata[B1], Any], np.complexfloating
        ],
        *,
        assert_unitary: bool = False,
        direction: Direction = "forward",
        data_id: uuid.UUID | None = None,
    ) -> None:
        super().__init__(
            cast("Any", matrix),
            assert_unitary=assert_unitary,
            direction=direction,
            data_id=data_id,
        )

    @property
    @override
    def eigenvectors(self) -> LegacyStateList[BasisMetadata, M]:
        """Get the eigenstates of the basis."""
        states = super().eigenvectors
        return build_legacy_state_list(states.basis, states.raw_data)
