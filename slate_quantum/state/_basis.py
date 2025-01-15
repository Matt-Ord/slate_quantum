from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast, override

import numpy as np
from slate import Array, Basis, SimpleMetadata
from slate.explicit_basis import ExplicitUnitaryBasis
from slate.metadata import BasisMetadata, Metadata2D

from slate_quantum.state._state import StateList

if TYPE_CHECKING:
    import uuid

    from slate.basis import BasisStateMetadata

type Direction = Literal["forward", "backward"]


class EigenstateBasis[
    M: BasisMetadata,
    B: Basis[Any, np.complexfloating] = Basis[M, np.complexfloating],
    BTransform: Basis[Any, Any] = Basis[
        Metadata2D[SimpleMetadata, BasisStateMetadata[B], None], Any
    ],
](ExplicitUnitaryBasis[M, np.complexfloating, B, BTransform]):
    """A basis with data stored as eigenstates."""

    def __init__[B1: Basis[Any, Any]](
        self: EigenstateBasis[Any, B1],
        matrix: Array[
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
    def eigenvectors(self) -> StateList[BasisMetadata, M]:
        """Get the eigenstates of the basis."""
        states = super().eigenvectors
        return StateList(states.basis, states.raw_data)
