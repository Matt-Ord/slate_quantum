from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, override

import numpy as np
from slate.basis import BasisFeature, WrappedBasis

from slate_quantum.operator._build._potential import RepeatedLengthMetadata

if TYPE_CHECKING:
    from collections.abc import Callable

    from slate.basis import Basis


class BlochShiftedBasis[
    DT: np.generic,
    M: RepeatedLengthMetadata,
    B: Basis[Any, Any] = Basis[M, DT],
](
    WrappedBasis[Any, DT, B],
):
    """A basis designed to show the underlying sparsity of the Bloch Hamiltonian.

    In the transformed basis, states in total momentum order according to

    k_tot = k_bloch + k_inner

    However if we instead order the basis so we can index using (inner_index, block_index)
    we can use a block diagonal basis to represent the Hamiltonian.

    This basis therefore deals with the necessary re-ordering of states for a
    transformed basis to be sparse under this representation.
    """

    def __init__[
        _B: Basis[Any, Any],
    ](
        self: BlochShiftedBasis[Any, Any, _B],
        inner: _B,
    ) -> None:
        super().__init__(cast("Any", inner))

    @property
    @override
    def inner(self) -> B:
        return self._inner

    @override
    def metadata(self) -> M:
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

        n_repeats = self.metadata().n_repeats
        n_states = self.metadata().inner.fundamental_size

        stacked = vectors.reshape(
            *vectors.shape[:axis], n_states, n_repeats, *vectors.shape[axis + 1 :]
        )
        # Order from smallest to largest bloch k
        shifted = np.fft.fftshift(stacked, axes=(axis, axis + 1))

        # This list is in order from smallest to largest
        # but we need to order according to fft conventions
        shift = n_repeats // 2 + (n_repeats * (n_states // 2))
        return np.roll(shifted.reshape(vectors.shape), -shift, axis=axis)

    @override
    def __from_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        axis %= vectors.ndim

        # The inner basis stores states in the
        # order k_inner + k_bloch
        n_repeats = self.metadata().n_repeats
        n_states = self.metadata().inner.fundamental_size
        # Apply a shift so we index like (k_inner, k_bloch)
        shift = n_repeats // 2 + (n_repeats * (n_states // 2))
        shifted = np.roll(vectors, shift, axis=axis)
        stacked = shifted.reshape(
            *vectors.shape[:axis], n_states, n_repeats, *vectors.shape[axis + 1 :]
        )
        # We want to index like (k_bloch, k_inner) using the
        # fourier indexing conventions
        return np.fft.ifftshift(stacked, axes=(axis, axis + 1)).reshape(vectors.shape)

    @override
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, BlochShiftedBasis)
            and (other.inner == self.inner)  # type: ignore unknown
            and self.is_dual == other.is_dual
        )

    @override
    def __hash__(self) -> int:
        return hash((3, self.inner, self.is_dual))

    @override
    def with_inner[  # type: ignore there is no way to bound inner in parent
        _B: Basis[Any, Any],
    ](self, inner: _B) -> BlochShiftedBasis[DT, M, _B]:
        return self.with_modified_inner(lambda _: inner)

    @override
    def with_modified_inner[  # type: ignore there is no way to bound the wrapper function in the parent class
        _DT: np.generic,
        _M: RepeatedLengthMetadata,
        _B: Basis[Any, Any] = Basis[_M, _DT],
    ](
        self,
        wrapper: Callable[[Basis[_M, _DT]], _B],
    ) -> BlochShiftedBasis[_DT, _M, _B]:
        """Get the wrapped basis after wrapper is applied to inner."""
        return BlochShiftedBasis[_DT, _M, _B](wrapper(self.inner))

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
