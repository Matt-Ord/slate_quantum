from __future__ import annotations

from typing import TYPE_CHECKING, Any, Never, cast, override

import numpy as np
from slate_core import Ctype
from slate_core.basis import AsUpcast, BasisConversion, BasisFeature, WrappedBasis

from slate_quantum._util.legacy import LegacyBasis
from slate_quantum.metadata import RepeatedLengthMetadata

if TYPE_CHECKING:
    from slate_core.basis import Basis

type LegacyBlochShiftedBasis[
    DT: np.generic,
    M: RepeatedLengthMetadata,
    B: Basis[Any, Any] = LegacyBasis[M, DT],
] = WrappedBasis[Any, Any]


class BlochShiftedBasis[
    B: Basis[RepeatedLengthMetadata] = Basis[RepeatedLengthMetadata],
    CT: Ctype[Never] = Ctype[Never],
](WrappedBasis[B, CT]):
    """A basis designed to show the underlying sparsity of the Bloch Hamiltonian.

    In the transformed basis, states in total momentum order according to

    k_tot = k_bloch + k_inner

    However if we instead order the basis so we can index using (inner_index, block_index)
    we can use a block diagonal basis to represent the Hamiltonian.

    This basis therefore deals with the necessary re-ordering of states for a
    transformed basis to be sparse under this representation.
    """

    def __init__[
        B_: Basis[RepeatedLengthMetadata, Any],
    ](
        self: BlochShiftedBasis[B_, Ctype[Never]],
        inner: B_,
    ) -> None:
        super().__init__(cast("Any", inner))

    @property
    @override
    def size(self) -> int:
        return self.inner.size

    @property
    @override
    def ctype(self) -> CT:
        return cast("CT", self.inner.ctype)

    @override
    def resolve_ctype[DT_: Ctype[Never]](
        self: BlochShiftedBasis[Basis[Any, DT_], Any],
    ) -> BlochShiftedBasis[B, DT_]:
        """Upcast the wrapped basis to a more specific type."""
        return cast("BlochShiftedBasis[B, DT_]", self)

    @override
    def upcast[M: RepeatedLengthMetadata](
        self: BlochShiftedBasis[Basis[M]],
    ) -> AsUpcast[BlochShiftedBasis[B, CT], M, CT]:
        return cast("Any", AsUpcast(self, self.metadata()))

    @override
    def __into_inner__[DT1: np.generic, DT2: np.generic, DT3: np.generic](
        self: WrappedBasis[Basis[Any, Ctype[DT3]], Ctype[DT1]],
        vectors: np.ndarray[Any, np.dtype[DT2]],
        axis: int = -1,
    ) -> BasisConversion[DT1, DT2, DT3]:
        axis %= vectors.ndim

        def fn() -> np.ndarray[Any, np.dtype[DT2]]:
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

        return BasisConversion(fn)

    @override
    def __from_inner__[DT1: np.generic, DT2: np.generic, DT3: np.generic](
        self: WrappedBasis[Basis[Any, Ctype[DT1]], Ctype[DT3]],
        vectors: np.ndarray[Any, np.dtype[DT2]],
        axis: int = -1,
    ) -> BasisConversion[DT1, DT2, DT3]:
        axis %= vectors.ndim

        def fn() -> np.ndarray[Any, np.dtype[DT2]]:
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
            return np.fft.ifftshift(stacked, axes=(axis, axis + 1)).reshape(
                vectors.shape
            )

        return BasisConversion(fn)

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
