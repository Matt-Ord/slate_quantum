from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast, overload, override

import numpy as np
from slate_core import Array, Basis, SimpleMetadata, ctype
from slate_core.explicit_basis import ExplicitUnitaryBasis
from slate_core.metadata import BasisMetadata, TupleMetadata

from slate_quantum.state._state import StateList

if TYPE_CHECKING:
    import uuid

    from slate_core.basis import BasisStateMetadata, TupleBasisLike

type Direction = Literal["forward", "backward"]


class EigenstateBasis[
    Transform: Array[
        Basis[
            TupleMetadata[tuple[SimpleMetadata, BasisStateMetadata[Basis]], None],
            ctype[np.complexfloating],
        ],
        np.dtype[np.complexfloating],
    ],
](ExplicitUnitaryBasis[Transform, ctype[np.complexfloating]]):
    """A basis with data stored as eigenstates."""

    @overload
    def __init__[
        Transform_: Array[
            Basis[
                TupleMetadata[tuple[SimpleMetadata, BasisStateMetadata[Basis]], None],
                ctype[np.complexfloating],
            ],
            np.dtype[np.complexfloating],
        ],
    ](
        self: ExplicitUnitaryBasis[Transform_],
        matrix: Transform_,
        *,
        direction: Literal["forward"] = "forward",
        data_id: uuid.UUID | None = None,
        assert_unitary: bool = False,
    ) -> None: ...

    @overload
    def __init__[
        M_: TupleMetadata[tuple[SimpleMetadata, BasisStateMetadata[Basis]], None],
        DT_: ctype[np.complexfloating],
    ](
        self: ExplicitUnitaryBasis[Array[Basis[M_, DT_], np.dtype[np.complexfloating]]],
        matrix: Array[Basis[M_, DT_], np.dtype[np.complexfloating]],
        *,
        direction: Literal["backward"],
        data_id: uuid.UUID | None = None,
        assert_unitary: bool = False,
    ) -> None: ...

    def __init__[B1: Basis[Any, Any]](
        self,
        matrix: Array[
            TupleBasisLike[tuple[BasisMetadata, BasisMetadata], None],
            np.dtype[np.number],
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

    @override
    def eigenvectors(self) -> StateList[BasisMetadata, M]:
        """Get the eigenstates of the basis."""
        states = super().eigenvectors()
        return StateList(states.basis, states.raw_data)
