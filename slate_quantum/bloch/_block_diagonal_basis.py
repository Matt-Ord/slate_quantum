from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, cast, override

import numpy as np
from slate.basis import BasisFeature, Padding, Truncation, TupleBasis2D, WrappedBasis
from slate.metadata import Metadata2D
from slate.util import pad_along_axis, truncate_along_axis

if TYPE_CHECKING:
    from collections.abc import Callable

    from slate.basis import Basis


def _spec_from_indices[DT: int | np.signedinteger](
    indices: tuple[DT, ...],
) -> str:
    return "".join(chr(cast("int", 97 + i)) for i in indices)


def extract_diagonal[DT: np.generic](
    array: np.ndarray[Any, np.dtype[DT]], axes: tuple[int, ...], out_axis: int = -1
) -> np.ndarray[Any, np.dtype[DT]]:
    input_indices = np.arange(array.ndim)
    input_indices[list(axes)] = input_indices[axes[0]]

    output_indices = np.delete(np.arange(array.ndim), list(axes))
    output_indices = np.insert(output_indices, out_axis, input_indices[axes[0]])

    square_slice = np.array([slice(None)] * array.ndim)
    n_out = np.min(np.array(array.shape)[list(axes)])
    square_slice[list(axes)] = slice(n_out)

    subscripts = f"{_spec_from_indices(tuple(input_indices))}->{_spec_from_indices(tuple(output_indices))}"
    return np.einsum(subscripts, array[tuple(square_slice)])  # type: ignore unknown


def build_diagonal[DT: np.generic](
    array: np.ndarray[Any, np.dtype[DT]],
    axis: int = -1,
    out_axes: tuple[int, ...] = (-1, -2),
    out_shape: tuple[int, ...] | None = None,
) -> np.ndarray[Any, np.dtype[DT]]:
    out_shape = out_shape or (array.shape[axis],) * len(out_axes)
    assert len(out_shape) == len(out_axes)
    n_dim_out = array.ndim - 1 + len(out_axes)

    eye_indices = np.mod(out_axes, n_dim_out)

    output_indices = np.arange(n_dim_out)

    input_indices = np.delete(np.arange(n_dim_out), list(out_axes))
    input_indices = np.insert(input_indices, axis, eye_indices[0])

    eye = np.zeros(out_shape, dtype=array.dtype)
    np.fill_diagonal(eye, 1)

    if array.shape[axis] == out_shape[0]:
        padded = array
    elif array.shape[axis] < out_shape[0]:
        padded = pad_along_axis(array, Padding(out_shape[0], 0, 0), axis)
    else:
        padded = truncate_along_axis(array, Truncation(out_shape[0], 0, 0), axis)

    subscripts = f"{_spec_from_indices(tuple(input_indices))},{_spec_from_indices(tuple(eye_indices))}->{_spec_from_indices(tuple(output_indices))}"
    return np.einsum(subscripts, padded, eye)  # type: ignore unknown


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
        axis %= vectors.ndim

        stacked = vectors.reshape(
            *vectors.shape[:axis],
            self.n_repeats,
            *self.block_shape,
            *vectors.shape[axis + 1 :],
        )
        return build_diagonal(
            stacked,
            axis,
            out_shape=self.repeat_shape,
            out_axes=tuple(range(axis, axis + 2 * len(self.block_shape), 2)),
        )

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
        inner_shape = tuple(
            (n_list, n_k) for (n_k, n_list) in zip(k_shape, list_shape, strict=False)
        )
        stacked = vectors.reshape(
            *vectors.shape[:axis],
            *(itertools.chain(*inner_shape)),
            *vectors.shape[axis + 1 :],
        )

        return extract_diagonal(
            stacked, tuple(range(axis, axis + 2 * len(inner_shape), 2)), axis
        )

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
