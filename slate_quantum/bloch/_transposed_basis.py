from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, cast, override

import numpy as np
from slate import StackedMetadata, TupleBasis
from slate.basis import BasisFeature, WrappedBasis
from slate.metadata import (
    AxisDirections,
)

from slate_quantum.metadata import RepeatedLengthMetadata

if TYPE_CHECKING:
    from collections.abc import Callable


class BlochTransposedBasis[
    DT: np.generic,
    M: RepeatedLengthMetadata,
    E: AxisDirections,
    B: TupleBasis[Any, Any, Any] = TupleBasis[M, E, DT],
](
    WrappedBasis[Any, DT, B],
):
    """A basis designed to show the underlying sparsity of the Bloch Hamiltonian.

    In the inner basis, states are indexed by [(inner_index, block_index), (inner_index, block_index), ...]

    this basis instead orders them according to [(block_index, block_index, ...), (inner_index, inner_index, ...)]

    This is so that they can play nice with BlockDiagonalBasis.
    """

    def __init__[
        _B: TupleBasis[Any, Any, Any],
    ](
        self: BlochTransposedBasis[Any, Any, Any, _B],
        inner: _B,
    ) -> None:
        super().__init__(cast("Any", inner))

    @property
    @override
    def inner(self) -> B:
        return self._inner

    @override
    def metadata(self) -> StackedMetadata[M, E]:
        """Metadata associated with the basis.

        Note: this should be a property, but this would ruin variance.
        """
        return self.inner.metadata()

    @property
    @override
    def size(self) -> int:
        return self.inner.size

    @override
    def __into_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        axis %= vectors.ndim

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

    @override
    def __from_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        axis %= vectors.ndim

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

    @override
    def with_inner[  # type: ignore there is no way to bound inner in parent
        _B: TupleBasis[Any, Any, Any],
    ](self, inner: _B) -> BlochTransposedBasis[DT, M, E, _B]:
        return self.with_modified_inner(lambda _: inner)

    @override
    def with_modified_inner[  # type: ignore there is no way to bound the wrapper function in the parent class
        _DT: np.generic,
        _M: RepeatedLengthMetadata,
        _E: AxisDirections,
        _B: TupleBasis[Any, Any, Any] = TupleBasis[_M, _E, _DT],
    ](
        self,
        wrapper: Callable[[TupleBasis[_M, _E, _DT]], _B],
    ) -> BlochTransposedBasis[_DT, _M, _E, _B]:
        """Get the wrapped basis after wrapper is applied to inner."""
        return BlochTransposedBasis[_DT, _M, _E, _B](wrapper(self.inner))

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
    def add_data[DT1: np.number[Any]](
        self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "LINEAR_MAP" not in self.features:
            msg = "add_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs + rhs).astype(lhs.dtype)

    @override
    def mul_data[DT1: np.number[Any]](
        self, lhs: np.ndarray[Any, np.dtype[DT1]], rhs: float
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "LINEAR_MAP" not in self.features:
            msg = "mul_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs * rhs).astype(lhs.dtype)

    @override
    def sub_data[DT1: np.number[Any]](
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
        return self.__from_inner__(self.inner.points)
