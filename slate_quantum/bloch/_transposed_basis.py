from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, Never, cast, overload, override

import numpy as np
from slate_core import Basis, Ctype, TupleBasis, TupleMetadata
from slate_core.basis import (
    AsUpcast,
    BasisConversion,
    BasisFeature,
    BlockDiagonalBasis,
    WrappedBasis,
)
from slate_core.metadata import AxisDirections

from slate_quantum.metadata import RepeatedLengthMetadata
from slate_quantum.operator._operator import OperatorMetadata

if TYPE_CHECKING:
    from slate_core.basis import Basis


class BlochTransposedBasis[
    B: TupleBasis[tuple[Basis[RepeatedLengthMetadata], ...], Any],
    CT: Ctype[Never] = Ctype[Never],
](
    WrappedBasis[B, CT],
):
    """A basis designed to show the underlying sparsity of the Bloch Hamiltonian.

    In the inner basis, states are indexed by [(inner_index, block_index), (inner_index, block_index), ...]

    this basis instead orders them according to [(block_index, block_index, ...), (inner_index, inner_index, ...)]

    This is so that they can play nice with BlockDiagonalBasis.
    """

    def __init__[B_: TupleBasis[tuple[Basis[RepeatedLengthMetadata], ...], Any]](
        self: BlochTransposedBasis[B_, Ctype[Never]], inner: B_
    ) -> None:
        super().__init__(cast("B", inner))

    @property
    @override
    def ctype(self) -> CT:
        return cast("CT", self.inner.ctype)

    @override
    def metadata[M_: RepeatedLengthMetadata, E](
        self: BlochTransposedBasis[TupleBasis[tuple[Basis[M_], ...], E], Any],
    ) -> TupleMetadata[tuple[M_, ...], E]:
        return self.inner.metadata()

    @override
    def resolve_ctype[DT_: Ctype[Never]](
        self: BlochTransposedBasis[TupleBasis[Any, Any, DT_], Any],
    ) -> BlochTransposedBasis[B, DT_]:
        """Upcast the wrapped basis to a more specific type."""
        return cast("BlochTransposedBasis[B, DT_]", self)

    @overload
    def upcast[M0: RepeatedLengthMetadata, E](
        self: BlochTransposedBasis[TupleBasis[tuple[Basis[M0]], E], Any],
    ) -> AsUpcast[BlochTransposedBasis[B, CT], TupleMetadata[tuple[M0], E], CT]: ...
    @overload
    def upcast[M0: RepeatedLengthMetadata, M1: RepeatedLengthMetadata, E](
        self: BlochTransposedBasis[TupleBasis[tuple[Basis[M0], Basis[M1]], E], Any],
    ) -> AsUpcast[BlochTransposedBasis[B, CT], TupleMetadata[tuple[M0, M1], E], CT]: ...
    @overload
    def upcast[
        M0: RepeatedLengthMetadata,
        M1: RepeatedLengthMetadata,
        M2: RepeatedLengthMetadata,
        E,
    ](
        self: BlochTransposedBasis[
            TupleBasis[tuple[Basis[M0], Basis[M1], Basis[M2]], E], Any
        ],
    ) -> AsUpcast[
        BlochTransposedBasis[B, CT], TupleMetadata[tuple[M0, M1, M2], E], CT
    ]: ...
    @overload
    def upcast[M: RepeatedLengthMetadata, E](
        self: BlochTransposedBasis[TupleBasis[tuple[Basis[M], ...], E], Any],
    ) -> AsUpcast[BlochTransposedBasis[B, CT], TupleMetadata[tuple[M, ...], E], CT]: ...
    @override
    def upcast(
        self,
    ) -> AsUpcast[
        BlochTransposedBasis[B, CT],
        TupleMetadata[tuple[RepeatedLengthMetadata, ...], Any],
        CT,
    ]:
        return cast("Any", AsUpcast(self, self.metadata()))

    @property
    @override
    def size(self) -> int:
        return self.inner.size

    @override
    def __into_inner__[DT1: np.generic, DT2: np.generic, DT3: np.generic](
        self: BlochTransposedBasis[
            TupleBasis[tuple[Basis[RepeatedLengthMetadata], ...], Ctype[DT3]],
            Ctype[DT1],
        ],
        vectors: np.ndarray[Any, np.dtype[DT2]],
        axis: int = -1,
    ) -> BasisConversion[DT1, DT2, DT3]:
        axis %= vectors.ndim

        def _fn() -> np.ndarray[Any, np.dtype[DT2]]:
            n_repeats = tuple(c.n_repeats for c in self.metadata().children)
            n_states = tuple(c.inner.fundamental_size for c in self.metadata().children)
            n_dim = len(n_repeats)

            # Indexed like [(block_index, block_index,...), (inner_index, inner_index,...)]
            stacked = vectors.reshape(
                *vectors.shape[:axis],
                *n_repeats,
                *n_states,
                *vectors.shape[axis + 1 :],
            )
            # Flip the vectors into [(inner_index, block_index),...]
            flipped = stacked.transpose(
                *range(axis),
                *itertools.chain(*((axis + i + n_dim, axis + i) for i in range(n_dim))),
                *range(axis + 2 * n_dim, stacked.ndim),
            )

            return flipped.reshape(vectors.shape)

        return BasisConversion(_fn)

    @override
    def __from_inner__[DT1: np.generic, DT2: np.generic, DT3: np.generic](
        self: BlochTransposedBasis[
            TupleBasis[tuple[Basis[RepeatedLengthMetadata], ...], Ctype[DT1]],
            Ctype[DT3],
        ],
        vectors: np.ndarray[Any, np.dtype[DT2]],
        axis: int = -1,
    ) -> BasisConversion[DT1, DT2, DT3]:
        axis %= vectors.ndim

        def _fn() -> np.ndarray[Any, np.dtype[DT2]]:
            n_repeats = tuple(c.n_repeats for c in self.metadata().children)
            n_states = tuple(c.inner.fundamental_size for c in self.metadata().children)
            n_dim = len(n_repeats)

            # Initially indexed like [(inner_index, block_index),...]
            stacked = vectors.reshape(
                *vectors.shape[:axis],
                *(itertools.chain(zip(n_states, n_repeats, strict=True))),
                *vectors.shape[axis + 1 :],
            )
            # Flip the vectors into basis [(block_index, block_index,...), (inner_index, inner_index,...)]
            flipped = stacked.transpose(
                *range(axis),
                *range(axis + 1, axis + 2 * n_dim, 2),
                *range(axis, axis + 2 * n_dim, 2),
                *range(axis + 2 * n_dim, stacked.ndim),
            )

            return flipped.reshape(vectors.shape)

        return BasisConversion(_fn)

    @override
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, BlochTransposedBasis)
            and (other.inner == self.inner)  # type: ignore unknown
            and self.is_dual == other.is_dual
        )

    @override
    def __hash__(self) -> int:
        return hash((3, self.inner, self.is_dual))

    @property
    @override
    def features(self) -> set[BasisFeature]:
        out = set[BasisFeature]()
        if "LINEAR_MAP" in self.inner.features:
            out.add("ADD")
            out.add("LINEAR_MAP")
            out.add("MUL")
            out.add("SUB")
        if "INDEX" in self.inner.features:
            out.add("INDEX")
        return out

    @override
    def add_data[DT1: np.number](
        self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "LINEAR_MAP" not in self.features:
            msg = "add_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs + rhs).astype(lhs.dtype)

    @override
    def mul_data[DT1: np.number](
        self, lhs: np.ndarray[Any, np.dtype[DT1]], rhs: complex
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "LINEAR_MAP" not in self.features:
            msg = "mul_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs * rhs).astype(lhs.dtype)

    @override
    def sub_data[DT1: np.number](
        self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "LINEAR_MAP" not in self.features:
            msg = "sub_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs - rhs).astype(lhs.dtype)

    @property
    @override
    def points(self) -> np.ndarray[Any, np.dtype[np.int_]]:
        if "INDEX" not in self.features:
            msg = "points not implemented for this basis"
            raise NotImplementedError(msg)

        return (
            cast("WrappedBasis[Any, Ctype[np.int_]]", self)
            .__from_inner__(self.inner.points)
            .ok()
        )


type BlochStateMetadata[
    M: RepeatedLengthMetadata = RepeatedLengthMetadata,
    E: AxisDirections = AxisDirections,
] = TupleMetadata[tuple[M, ...], E]

type BlochStateBasis[
    M: RepeatedLengthMetadata = RepeatedLengthMetadata,
    E: AxisDirections = AxisDirections,
] = AsUpcast[
    BlochTransposedBasis[TupleBasis[tuple[Basis[M], ...], E]],
    BlochStateMetadata[M, E],
]

type BlochOperatorBasis[
    M: RepeatedLengthMetadata = RepeatedLengthMetadata,
    E: AxisDirections = AxisDirections,
] = AsUpcast[
    BlockDiagonalBasis[
        TupleBasis[
            tuple[BlochStateBasis[M, E], BlochStateBasis[M, E]],
            None,
        ]
    ],
    OperatorMetadata[BlochStateMetadata[M, E]],
]
