from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, cast, override

import numpy as np
from slate.basis import BasisFeature, TupleBasis2D, WrappedBasis
from slate.metadata import Metadata2D

if TYPE_CHECKING:
    from collections.abc import Callable

    from slate.basis import Basis


class BlockDiagonalBasis[
    DT: np.generic,
    B0: Basis[Any, Any],
    B1: Basis[Any, Any],
    E,
](
    WrappedBasis[Metadata2D[Any, Any, E], DT, TupleBasis2D[DT, Any, Any, E]],
):
    """Represents a diagonal basis."""

    def __init__[_DT: np.generic, _B0: Basis[Any, Any], _B1: Basis[Any, Any], _E](
        self: BlockDiagonalBasis[_DT, _B0, _B1, _E],
        inner: TupleBasis2D[_DT, _B0, _B1, _E],
        block_shape: tuple[int, int],
    ) -> None:
        super().__init__(cast("Any", inner))
        assert self.inner.children[0].size % block_shape[0] == 0
        assert self.inner.children[1].size % block_shape[1] == 0
        self._block_shape = block_shape

    @property
    @override
    def inner(self) -> TupleBasis2D[DT, B0, B1, E]:
        return cast("TupleBasis2D[DT, B0, B1, E]", self._inner)

    @property
    def block_shape(self) -> tuple[int, int]:
        """The shape of each block matrix along the diagonal."""
        return self._block_shape

    @property
    def repeat_shape(self) -> tuple[int, int]:
        """The shape of the repeats of blocks."""
        return (
            self.inner.children[0].size // self.block_shape[0],
            self.inner.children[1].size // self.block_shape[1],
        )

    @property
    def n_repeats(self) -> int:
        """Total number of repeats."""
        return min(self.repeat_shape)

    @property
    @override
    def size(self) -> int:
        return np.prod(self.block_shape).item() * self.n_repeats

    @override
    def __into_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        swapped = vectors.swapaxes(axis, 0)
        stacked = swapped.reshape(self.n_repeats, *self.block_shape, *swapped.shape[1:])
        stacked = np.moveaxis(stacked, 0, 2)

        out = np.zeros(
            (*self.block_shape, np.prod(self.repeat_shape).item(), *swapped.shape[1:]),
            dtype=stacked.dtype,
        )
        for i in range(self.n_repeats):
            out[:, :, i * (self.repeat_shape[0] + 1)] = stacked[:, :, i]
        out = np.moveaxis(out, 2, 0)

        return out.reshape(self.inner.size, *swapped.shape[1:]).swapaxes(axis, 0)

    # @override
    # def __from_inner__[DT1: np.generic](  # [DT1: DT]
    #     self,
    #     vectors: np.ndarray[Any, np.dtype[DT1]],
    #     axis: int = -1,
    # ) -> np.ndarray[Any, np.dtype[DT1]]:
    #     swapped = vectors.swapaxes(axis, 0)
    #     stacked = swapped.reshape(
    #         self.block_shape[0], self.block_shape[1], -1, *swapped.shape[1:]
    #     )
    #     # TODO: this is actually complicated - need to copy from the old impl...
    #     out = np.zeros(
    #         (*self.block_shape, self.n_repeats, *swapped.shape[1:]), dtype=stacked.dtype
    #     )
    #     for i in range(self.n_repeats):
    #         print(i, self.repeat_shape, i * (self.repeat_shape[1] + 1))
    #         print(stacked[:, i * (self.repeat_shape[1] + 1)])
    #         # out[:, i] = stacked[:, i * (self.repeat_shape[1] + 1)]

    #     return out.reshape(self.size, *swapped.shape[1:]).swapaxes(axis, 0)

    @override
    def __from_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        axis %= vectors.ndim

        # The vectors in the inner basis are stored in
        # the shape [(list,momentum), (list,momentum),...]
        k_shape = self.block_shape
        list_shape = self.repeat_shape
        print(k_shape, list_shape)

        inner_shape = tuple(
            (n_list, n_k) for (n_k, n_list) in zip(k_shape, list_shape, strict=False)
        )

        shifted = vectors.reshape(
            *vectors.shape[:axis],
            *(itertools.chain(*inner_shape)),
            *vectors.shape[axis + 1 :],
        )
        print(shifted.shape)
        print(shifted[0, :, 0])

        # Flip the vectors into basis [(list, list, ...), (momentum, momentum, ...)]
        flipped = shifted.transpose(
            *range(axis),
            *range(axis, axis + 2 * len(inner_shape), 2),
            *range(axis + 1, axis + 2 * len(inner_shape), 2),
            *range(axis + 2 * len(inner_shape), shifted.ndim),
        )
        # TODO: N dimensional diag ...

        return np.moveaxis(np.diagonal(flipped, axis1=axis, axis2=axis + 1), -1, axis)

    @override
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, BlockDiagonalBasis)
            and (other.inner == self.inner)  # type: ignore unknown
            and self.is_dual == other.is_dual
        )

    @override
    def __hash__(self) -> int:
        return hash((3, self.inner, self.is_dual))

    @override
    def with_inner[  # type: ignore there is no way to bound inner in parent
        DT1: np.generic,
        B01: Basis[Any, Any],
        B11: Basis[Any, Any],
        E1,
    ](
        self, inner: TupleBasis2D[DT1, B01, B11, E1]
    ) -> BlockDiagonalBasis[DT1, B01, B11, E1]:
        return self.with_modified_inner(lambda _: inner)

    @override
    def with_modified_inner[  # type: ignore there is no way to bound the wrapper function in the parent class
        DT1: np.generic,
        B01: Basis[Any, Any],
        B11: Basis[Any, Any],
        E1,
    ](
        self,
        wrapper: Callable[
            [TupleBasis2D[DT, Any, Any, E]], TupleBasis2D[DT1, B01, B11, E1]
        ],
    ) -> BlockDiagonalBasis[DT1, B01, B11, E1]:
        """Get the wrapped basis after wrapper is applied to inner."""
        return BlockDiagonalBasis[DT1, B01, B11, E1](
            wrapper(self.inner), self.block_shape
        )

    @property
    @override
    def features(self) -> set[BasisFeature]:
        out = set[BasisFeature]()
        if "SIMPLE_ADD" in self.inner.features:
            out.add("ADD")
            out.add("SIMPLE_ADD")
        if "SIMPLE_MUL" in self.inner.features:
            out.add("MUL")
            out.add("SIMPLE_MUL")
        if "SIMPLE_SUB" in self.inner.features:
            out.add("SUB")
            out.add("SIMPLE_SUB")
        if "INDEX" in self.inner.features:
            out.add("INDEX")
        return out

    @override
    def add_data[DT1: np.number[Any]](
        self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "SIMPLE_ADD" not in self.features:
            msg = "add_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs + rhs).astype(lhs.dtype)

    @override
    def mul_data[DT1: np.number[Any]](
        self, lhs: np.ndarray[Any, np.dtype[DT1]], rhs: float
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "SIMPLE_MUL" not in self.features:
            msg = "mul_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs * rhs).astype(lhs.dtype)

    @override
    def sub_data[DT1: np.number[Any]](
        self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "SIMPLE_SUB" not in self.features:
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
