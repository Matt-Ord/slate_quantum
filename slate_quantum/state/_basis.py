from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Never, cast, overload, override

import numpy as np
from slate_core import Array, Basis, Ctype, SimpleMetadata
from slate_core.explicit_basis import ExplicitUnitaryBasis
from slate_core.metadata import BasisMetadata, TupleMetadata

from slate_quantum.state._state import StateList

if TYPE_CHECKING:
    import uuid

    from slate_core.basis import (
        AsUpcast,
        BasisStateMetadata,
        RecastBasis,
        TupleBasis2D,
        TupleBasisLike,
    )


type Direction = Literal["forward", "backward"]


class EigenstateBasis[
    Transform: Array[
        Basis[TupleMetadata[tuple[SimpleMetadata, BasisStateMetadata[Basis]], None]],
        np.dtype[np.number],
    ],
](ExplicitUnitaryBasis[Transform, Ctype[np.complexfloating]]):
    """A basis with data stored as eigenstates."""

    @overload
    def __init__[
        Transform_: Array[
            Basis[
                TupleMetadata[tuple[SimpleMetadata, BasisStateMetadata[Basis]], None]
            ],
            np.dtype[np.number],
        ],
    ](
        self: EigenstateBasis[Transform_],
        matrix: Transform_,
        *,
        direction: Literal["forward"] = "forward",
        data_id: uuid.UUID | None = None,
        assert_unitary: bool = False,
    ) -> None: ...

    @overload
    def __init__[
        M_: TupleMetadata[tuple[SimpleMetadata, BasisStateMetadata[Basis]], None],
        DT_: Ctype[np.complexfloating],
    ](
        self: EigenstateBasis[Array[Basis[M_, DT_], np.dtype[np.number]]],
        matrix: Array[Basis[M_, DT_], np.dtype[np.number]],
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
    def eigenvectors[
        M1_: SimpleMetadata,
        BInner_: Basis,
        DT_: np.dtype[np.complexfloating],
        DT1_: Ctype[Never],
    ](
        self: EigenstateBasis[
            Array[
                Basis[
                    TupleMetadata[tuple[M1_, BasisStateMetadata[BInner_]], None], DT1_
                ],
                DT_,
            ]
        ],
    ) -> StateList[
        AsUpcast[
            RecastBasis[
                TupleBasis2D[tuple[Basis[M1_, Ctype[np.generic]], BInner_], None],
                TupleBasisLike[
                    tuple[M1_, BasisStateMetadata[BInner_]], None, Ctype[np.generic]
                ],
                TupleBasisLike[tuple[M1_, BasisStateMetadata[BInner_]], None, DT1_],
            ],
            TupleMetadata[tuple[M1_, BasisMetadata], None],
            Any,
        ],
        np.dtype[np.complexfloating],
    ]:
        states = super().eigenvectors()
        return StateList(states.basis, states.data.astype(np.complex128))  # type: ignore[return-value]


type EigenstateBasisWithInner[Inner: Basis] = EigenstateBasis[
    Array[
        Basis[TupleMetadata[tuple[SimpleMetadata, BasisStateMetadata[Inner]], None]],
        np.dtype[np.complexfloating],
    ],
]
