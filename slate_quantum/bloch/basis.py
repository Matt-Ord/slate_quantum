from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, override

import numpy as np
from slate.metadata import BasisMetadata, LabelSpacing
from slate.metadata.length import SpacedLengthMetadata

from slate_quantum.state import EigenstateBasis, StateList

if TYPE_CHECKING:
    from slate.basis import Basis


class BlochLengthMetadata(SpacedLengthMetadata):
    def __init__(self, inner: SpacedLengthMetadata, n_repeats: int) -> None:
        self._inner = inner
        self.n_repeats = n_repeats
        super().__init__(
            inner.fundamental_size * n_repeats,
            spacing=LabelSpacing(
                start=inner.spacing.start, delta=n_repeats * inner.spacing.delta
            ),
        )

    @property
    def inner(self) -> SpacedLengthMetadata:
        return self._inner


type BlochWavefunctionList[Any] = StateList[Any]
"""Represents a list of wavefunctions in the reduced bloch basis, indexed by bloch k."""


type BlochWavefunctionListList[Any] = StateList[Any]
"""Represents a list of wavefunctions, indexed by band and bloch k."""


def get_wavepacket_basis(
    basis: BlochWavefunctionListListBasis[_B0, _SB0, _SB1],
) -> BlochWavefunctionListBasis[_SB0, _SB1]:
    """
    Get the basis of the wavepacket.

    Parameters
    ----------
    wavepackets : WavepacketList[_B0, _SB0, _SB1]

    Returns
    -------
    WavepacketBasis[_SB0, _SB1]
    """
    return VariadicTupleBasis((basis[0][1], basis[1]), None)


def get_fundamental_wavepacket(
    wavepackets: BlochWavefunctionListList[_B0, _SB0, _SBV0],
) -> BlochWavefunctionListList[
    _B0,
    TupleBasisLike[*tuple[FundamentalTransformedBasis, ...]],
    TupleBasisWithLengthLike[*tuple[FundamentalTransformedPositionBasis, ...]],
]:
    """Get the wavepacket in the fundamental (transformed) basis."""
    basis = _get_fundamental_wavepacket_basis(wavepackets)
    converted_basis = basis[1]
    converted_list_basis = basis[0]
    return convert_wavepacket_to_basis(
        wavepackets,
        basis=converted_basis,
        list_basis=VariadicTupleBasis(
            (wavepackets["basis"][0][0], converted_list_basis), None
        ),
    )


def get_bloch_states(
    wavepackets: BlochWavefunctionListList[_B0, _SB0, _SBV0],
) -> LegacyStateVectorList[
    TupleBasisLike[_B0, TupleBasisLike[*tuple[FundamentalTransformedBasis, ...]]],
    TupleBasisWithLengthLike[*tuple[FundamentalTransformedPositionBasis, ...]],
]:
    """
    Uncompress bloch wavefunction list.

    A bloch wavefunction list is implicitly compressed, as each wavefunction in the list
    only stores the state at the relevant non-zero bloch k. This function undoes this implicit
    compression

    Parameters
    ----------
    wavepacket : BlochWavefunctionListList[_B0, _SB0, _SB1]
        The wavepacket to decompress

    Returns
    -------
    StateVectorList[
    TupleBasisLike[_B0, _SB0],
    TupleBasisLike[*tuple[FundamentalTransformedPositionBasis, ...]],
    ]
    """
    converted = get_fundamental_wavepacket(wavepackets)

    decompressed_basis = TupleBasis(
        *tuple(
            FundamentalTransformedPositionBasis(b1.delta_x * b0.n, b0.n * b1.n)
            for (b0, b1) in zip(
                converted["basis"][0][1], converted["basis"][1], strict=True
            )
        )
    )
    out = np.zeros(
        (*wavepackets["basis"][0].shape, decompressed_basis.n), dtype=np.complex128
    )

    util = BasisUtil(converted["basis"][0][1])
    # for each bloch k
    for idx in range(converted["basis"][0][1].n):
        offset = util.get_stacked_index(idx)

        states = _get_compressed_bloch_states_at_bloch_idx(converted, idx)
        # Re-interpret as a sampled state, and convert to a full state
        full_states = convert_state_vector_list_to_basis(
            {
                "basis": TupleBasis(
                    states["basis"][0],
                    _get_sampled_basis(
                        VariadicTupleBasis(
                            (converted["basis"][0][1], converted["basis"][1]), None
                        ),
                        offset,
                    ),
                ),
                "data": states["data"],
            },
            decompressed_basis,
        )

        out[:, idx, :] = full_states["data"].reshape(
            wavepackets["basis"][0][0].n, decompressed_basis.n
        )

    return {
        "basis": VariadicTupleBasis((converted["basis"][0], decompressed_basis), None),
        "data": out.ravel(),
    }


class BlochEigenstateBasis[M: BasisMetadata](EigenstateBasis[M]):
    def __init__(
        self,
        wavefunctions: BlochWavefunctionListList[_B0, _SB0, _SBV0],
    ) -> None:
        converted_list_basis = tuple_basis_as_transformed_fundamental(
            wavefunctions["basis"][0][1]
        )
        self._wavefunctions = convert_wavepacket_to_basis(
            wavefunctions,
            list_basis=VariadicTupleBasis(
                (wavefunctions["basis"][0][0], converted_list_basis), None
            ),
        )

    @cached_property
    def bloch_states(
        self,
    ) -> StateList[
        VariadicTupleBasis[
            np.complex128,
            Basis[BasisMetadata, np.generic],
            Basis[M, np.complex128],
            None,
        ]
    ]:
        return get_bloch_states(self.wavefunctions)

    @cached_property
    def states(
        self,
    ) -> StateList[
        VariadicTupleBasis[
            np.complex128,
            Basis[BasisMetadata, np.generic],
            Basis[M, np.complex128],
            None,
        ]
    ]:
        return get_bloch_states(self.wavefunctions)

    @override
    def __into_inner__[DT1: np.complex128](
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        axis %= vectors.ndim

        vectors_in_basis = np.apply_along_axis(
            lambda x: np.einsum(  # type: ignore lib
                "ij,ijk->jk",
                x.reshape(
                    self.wavefunctions["basis"][0].shape,
                ),
                self.wavefunctions["data"].reshape(
                    *self.wavefunctions["basis"][0].shape,
                    self.wavefunctions["basis"][1].size,
                ),
            ).ravel(),
            axis,
            vectors,
        )
        # The vectors in basis [(list, list, ...), (momentum, momentum, ...)]
        converted = convert_vector(
            vectors_in_basis,
            get_wavepacket_basis(self.wavefunctions["basis"]),
            _get_fundamental_wavepacket_basis(self.wavefunctions),
            axis=axis,
        )

        k_shape = self.wavefunctions["basis"][1].shape
        list_shape = self.wavefunctions["basis"][0][1].shape
        ndim = len(k_shape)

        inner_shape = tuple(zip(list_shape, k_shape, strict=False))
        shifted = np.fft.fftshift(
            converted.reshape(
                *vectors_in_basis.shape[:axis],
                *tuple(
                    item for pair in zip(*inner_shape, strict=False) for item in pair
                ),
                *vectors_in_basis.shape[axis + 1 :],
            ),
            axes=tuple(range(axis, axis + 2 * ndim)),
        )

        # Convert the vectors into fundamental unfurled basis
        # ie [(momentum,list), (momentum,list),...
        transposed = shifted.transpose(
            *range(axis),
            *itertools.chain(*((axis + i + ndim, axis + i) for i in range(ndim))),
            *range(axis + 2 * len(inner_shape), shifted.ndim),
        )

        # Shift the 0 frequency back to the start and flatten
        # Note the ifftshift would shift by (list_shape[i] * states_shape[i]) // 2
        # Which is wrong in this case
        shift = tuple(
            -(n_list // 2 + (n_list * (n_s // 2))) for (n_list, n_s) in inner_shape
        )
        unshifted = np.roll(
            transposed.reshape(
                *vectors_in_basis.shape[:axis],
                *tuple(np.prod(x) for x in inner_shape),
                *vectors_in_basis.shape[axis + 1 :],
            ),
            shift,
            tuple(range(axis, axis + ndim)),
        )
        if self.vectors_basis.n < 100 * 90:
            np.testing.assert_allclose(
                self.vectors_basis[1].__into_fundamental__(
                    unshifted.reshape(vectors_in_basis.shape), axis
                ),
                ExplicitStackedBasisWithLength(self.vectors).__into_fundamental__(
                    vectors, axis
                ),
                atol=1e-10,
            )
        return unshifted

    @override
    def __from_inner__[DT1: np.complex128](
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        axis %= vectors.ndim
        # The vectors in fundamental unfurled basis [(momentum,list), (momentum,list),...]

        k_shape = self.wavefunctions["basis"][1].fundamental_shape
        list_shape = self.wavefunctions["basis"][0][1].fundamental_shape

        inner_shape = tuple(
            (n_k, n_list) for (n_k, n_list) in zip(k_shape, list_shape, strict=False)
        )

        shift = tuple(
            (n_list // 2 + (n_list * (n_s // 2))) for (n_s, n_list) in inner_shape
        )
        shifted = np.roll(
            vectors_in_basis.reshape(
                *vectors_in_basis.shape[:axis],
                *(np.prod(x) for x in inner_shape),
                *vectors_in_basis.shape[axis + 1 :],
            ),
            shift,
            axis=tuple(range(axis, axis + len(inner_shape))),
        ).reshape(
            *vectors_in_basis.shape[:axis],
            *(itertools.chain(*inner_shape)),
            *vectors_in_basis.shape[axis + 1 :],
        )

        # Flip the vectors into basis [(list, list, ...), (momentum, momentum, ...)]
        flipped = shifted.transpose(
            *range(axis),
            *range(axis + 1, axis + 2 * len(inner_shape), 2),
            *range(axis, axis + 2 * len(inner_shape), 2),
            *range(axis + 2 * len(inner_shape), shifted.ndim),
        )
        unshifted = np.fft.ifftshift(
            flipped,
            axes=tuple(range(axis, axis + 2 * len(inner_shape))),
        )
        # Convert from fundamental to wavepacket basis
        converted = convert_array(
            unshifted.reshape(vectors_in_basis.shape),
            _get_fundamental_wavepacket_basis(self.wavefunctions),
            get_wavepacket_basis(self.wavefunctions["basis"]),
            axis=axis,
        )

        if self.vectors_basis.n < 100 * 90:
            np.testing.assert_allclose(
                np.apply_along_axis(
                    lambda x: np.einsum(  # type: ignore lib
                        "jk,ijk->ij",
                        x.reshape(
                            self.wavefunctions["basis"][0][1].n,
                            self.wavefunctions["basis"][1].size,
                        ),
                        np.conj(
                            self.wavefunctions["data"].reshape(
                                *self.wavefunctions["basis"][0].shape,
                                self.wavefunctions["basis"][1].size,
                            )
                        ),
                    ).ravel(),
                    axis,
                    converted,
                ),
                ExplicitStackedBasisWithLength(self.vectors).__from_fundamental__(
                    vectors, axis
                ),
                atol=1e-10,
            )

        return np.apply_along_axis(
            lambda x: np.einsum(  # type: ignore lib
                "jk,ijk->ij",
                x.reshape(
                    self.wavefunctions["basis"][0][1].n,
                    self.wavefunctions["basis"][1].size,
                ),
                np.conj(
                    self.wavefunctions["data"].reshape(
                        *self.wavefunctions["basis"][0].shape,
                        self.wavefunctions["basis"][1].size,
                    )
                ),
            ).ravel(),
            axis,
            converted,
        )

    def __into_fundamental__(
        self,
        vectors: np.ndarray[
            _S0Inv,
            np.dtype[np.complex128]
            | np.dtype[np.float64]
            | np.dtype[np.float64 | np.complex128],
        ],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
        axis %= vectors.ndim

        vectors_in_basis = np.apply_along_axis(
            lambda x: np.einsum(  # type: ignore lib
                "ij,ijk->jk",
                x.reshape(
                    self.wavefunctions["basis"][0].shape,
                ),
                self.wavefunctions["data"].reshape(
                    *self.wavefunctions["basis"][0].shape,
                    self.wavefunctions["basis"][1].size,
                ),
            ).ravel(),
            axis,
            vectors,
        )
        # The vectors in basis [(list, list, ...), (momentum, momentum, ...)]
        converted = convert_vector(
            vectors_in_basis,
            get_wavepacket_basis(self.wavefunctions["basis"]),
            _get_fundamental_wavepacket_basis(self.wavefunctions),
            axis=axis,
        )

        k_shape = self.wavefunctions["basis"][1].shape
        list_shape = self.wavefunctions["basis"][0][1].shape
        ndim = len(k_shape)

        inner_shape = tuple(zip(list_shape, k_shape, strict=False))
        shifted = np.fft.fftshift(
            converted.reshape(
                *vectors_in_basis.shape[:axis],
                *tuple(
                    item for pair in zip(*inner_shape, strict=False) for item in pair
                ),
                *vectors_in_basis.shape[axis + 1 :],
            ),
            axes=tuple(range(axis, axis + 2 * ndim)),
        )

        # Convert the vectors into fundamental unfurled basis
        # ie [(momentum,list), (momentum,list),...
        transposed = shifted.transpose(
            *range(axis),
            *itertools.chain(*((axis + i + ndim, axis + i) for i in range(ndim))),
            *range(axis + 2 * len(inner_shape), shifted.ndim),
        )

        # Shift the 0 frequency back to the start and flatten
        # Note the ifftshift would shift by (list_shape[i] * states_shape[i]) // 2
        # Which is wrong in this case
        shift = tuple(
            -(n_list // 2 + (n_list * (n_s // 2))) for (n_list, n_s) in inner_shape
        )
        unshifted = np.roll(
            transposed.reshape(
                *vectors_in_basis.shape[:axis],
                *tuple(np.prod(x) for x in inner_shape),
                *vectors_in_basis.shape[axis + 1 :],
            ),
            shift,
            tuple(range(axis, axis + ndim)),
        )
        if self.vectors_basis.n < 100 * 90:
            np.testing.assert_allclose(
                self.vectors_basis[1].__into_fundamental__(
                    unshifted.reshape(vectors_in_basis.shape), axis
                ),
                ExplicitStackedBasisWithLength(self.vectors).__into_fundamental__(
                    vectors, axis
                ),
                atol=1e-10,
            )

        return self.vectors_basis[1].__into_fundamental__(
            unshifted.reshape(vectors_in_basis.shape), axis
        )

    @property
    def vectors_basis(
        self: Self,
    ) -> TupleBasis[
        TupleBasisLike[_B0, TupleBasisLike[*tuple[FundamentalTransformedBasis, ...]]],
        TupleBasisWithLengthLike[*tuple[FundamentalTransformedPositionBasis, ...]],
    ]:
        """Get the basis of the full states list."""
        return TupleBasis(
            self.wavefunctions["basis"][0],
            self.unfurled_basis,
        )

    @property
    def unfurled_basis(
        self: Self,
    ) -> TupleBasisWithLengthLike[*tuple[FundamentalTransformedPositionBasis, ...]]:
        """Get the basis of the full states."""
        return get_fundamental_unfurled_basis(
            get_wavepacket_basis(self.wavefunctions["basis"])
        )

    @property
    def wavefunctions(
        self: Self,
    ) -> BlochWavefunctionListList[
        _B0,
        TupleBasisLike[*tuple[FundamentalTransformedBasis, ...]],
        StackedBasisWithVolumeLike,
    ]:
        """Get the raw wavefunctions."""
        return self._wavefunctions

    def vectors_at_bloch_k(
        self: Self, idx: SingleIndexLike
    ) -> LegacyStateVectorList[
        _B0,
        StackedBasisWithVolumeLike,
    ]:
        """Get the states at bloch idx."""
        util = BasisUtil(self.wavefunctions["basis"][0][1])
        idx = util.get_flat_index(idx, mode="wrap") if isinstance(idx, tuple) else idx
        data = self.wavefunctions["data"].reshape(
            *self.wavefunctions["basis"][0].shape, -1
        )[:, idx, :]
        return {
            "basis": TupleBasis(
                self.wavefunctions["basis"][0][0], self.wavefunctions["basis"][1]
            ),
            "data": data.ravel(),
        }

    def basis_at_bloch_k(
        self: Self, idx: tuple[int, ...]
    ) -> ExplicitStackedBasisWithLength[
        _B0,
        StackedBasisWithVolumeLike,
    ]:
        """Get the basis at bloch idx."""
        return ExplicitStackedBasisWithLength(self.vectors_at_bloch_k(idx))
