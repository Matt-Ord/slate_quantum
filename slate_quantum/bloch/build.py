from __future__ import annotations

from itertools import starmap
from typing import TYPE_CHECKING, Any, override

import numpy as np
from slate import StackedMetadata, basis, tuple_basis
from slate import metadata as _metadata
from slate.basis import BlockDiagonalBasis
from slate.metadata import (
    AxisDirections,
    LabeledMetadata,
    Metadata2D,
    SpacedLengthMetadata,
    SpacedVolumeMetadata,
)

from slate_quantum import operator
from slate_quantum.bloch._shifted_basis import BlochShiftedBasis
from slate_quantum.bloch._transposed_basis import BlochTransposedBasis
from slate_quantum.metadata import RepeatedLengthMetadata
from slate_quantum.operator._operator import Operator, OperatorList

if TYPE_CHECKING:
    from slate.basis import TupleBasis2D

    from slate_quantum.operator._diagonal import Potential


class BlochFractionMetadata(LabeledMetadata[np.floating]):
    """Metadata for the Bloch fraction."""

    def __init__(self, size: int) -> None:
        super().__init__(size)

    @property
    @override
    def values(self) -> np.ndarray[Any, np.dtype[np.floating]]:
        """Shape of the full data."""
        return _metadata.fundamental_nk_points(self) / self.fundamental_size

    @staticmethod
    def from_repeats(
        repeat: tuple[int, ...],
    ) -> StackedMetadata[BlochFractionMetadata, None]:
        """Build a stacked metadata from a tuple of repeats."""
        return StackedMetadata(tuple(BlochFractionMetadata(n) for n in repeat), None)


def metadata_from_split(
    fraction_meta: StackedMetadata[BlochFractionMetadata, None],
    state_meta: SpacedVolumeMetadata,
) -> StackedMetadata[RepeatedLengthMetadata, AxisDirections]:
    """Get the metadata for the Bloch operator."""
    return StackedMetadata(
        tuple(
            starmap(
                RepeatedLengthMetadata,
                zip(state_meta.children, fraction_meta.shape, strict=True),
            )
        ),
        state_meta.extra,
    )


def basis_from_metadata(
    metadata: StackedMetadata[RepeatedLengthMetadata, AxisDirections],
) -> BlochTransposedBasis[
    np.complexfloating,
    RepeatedLengthMetadata,
    AxisDirections,
]:
    """Build the Bloch basis."""
    operator_basis = basis.fundamental_transformed_tuple_basis_from_metadata(metadata)
    return BlochTransposedBasis(
        basis.with_modified_children(operator_basis, lambda _, i: BlochShiftedBasis(i))
    )


def _metadata_from_operator_list(
    meta: Metadata2D[
        StackedMetadata[BlochFractionMetadata, None],
        Metadata2D[SpacedVolumeMetadata, SpacedVolumeMetadata, None],
        None,
    ],
) -> BlochTransposedBasis[
    np.complexfloating,
    RepeatedLengthMetadata,
    AxisDirections,
]:
    """Get the metadata for the Bloch operator."""
    list_meta = meta[0]
    single_operator_meta = meta[1][0]
    full_operator_metadata = metadata_from_split(list_meta, single_operator_meta)
    return basis_from_metadata(full_operator_metadata)


def bloch_operator_from_list[
    M0: StackedMetadata[BlochFractionMetadata, None],
    M1: SpacedVolumeMetadata,
](
    operators: OperatorList[M0, M1, np.complexfloating],
) -> Operator[
    StackedMetadata[RepeatedLengthMetadata, AxisDirections],
    np.complexfloating,
    BlockDiagonalBasis[
        np.complexfloating,
        RepeatedLengthMetadata,
        AxisDirections,
        TupleBasis2D[
            np.complexfloating,
            BlochTransposedBasis[
                np.complexfloating,
                RepeatedLengthMetadata,
                AxisDirections,
            ],
            BlochTransposedBasis[
                np.complexfloating,
                RepeatedLengthMetadata,
                AxisDirections,
            ],
            None,
        ],
    ],
]:
    """Build the block diagonal Bloch Hamiltonian from a list of operators."""
    operators = operators.with_list_basis(
        basis.from_metadata(operators.basis.metadata()[0])
    )
    operators = operators.with_operator_basis(
        basis.fundamental_transformed_tuple_basis_from_metadata(
            operators.basis[1].metadata(), is_dual=operators.basis[1].is_dual
        )
    )

    operator_basis = _metadata_from_operator_list(operators.basis.metadata())
    out_basis = BlockDiagonalBasis(
        tuple_basis((operator_basis, operator_basis.dual_basis())),
        operators.basis.metadata()[1].shape,
    )
    return Operator(out_basis, operators.raw_data)


def _get_sample_fractions[M: BlochFractionMetadata](
    metadata: StackedMetadata[M, None],
) -> tuple[np.ndarray[tuple[int], np.dtype[np.floating]], ...]:
    """Get the frequencies of the samples in a wavepacket, as a fraction of dk."""
    mesh = np.meshgrid(*(n.values for n in metadata), indexing="ij")
    return tuple(nki.ravel() for nki in mesh)


def kinetic_hamiltonian[M: SpacedLengthMetadata, E: AxisDirections](
    potential: Potential[M, E, np.complexfloating],
    mass: float,
    repeat: tuple[int, ...],
) -> Operator[
    StackedMetadata[RepeatedLengthMetadata, AxisDirections],
    np.complexfloating,
    BlockDiagonalBasis[
        np.complexfloating,
        RepeatedLengthMetadata,
        AxisDirections,
        TupleBasis2D[
            np.complexfloating,
            BlochTransposedBasis[
                np.complexfloating,
                RepeatedLengthMetadata,
                AxisDirections,
            ],
            BlochTransposedBasis[
                np.complexfloating,
                RepeatedLengthMetadata,
                AxisDirections,
            ],
            None,
        ],
    ],
]:
    """Build a kinetic Hamiltonian in the Bloch basis."""
    list_metadata = BlochFractionMetadata.from_repeats(repeat)
    bloch_fractions = _get_sample_fractions(list_metadata)
    list_basis = basis.from_metadata(list_metadata)

    operators = OperatorList.from_operators(
        [
            operator.build.kinetic_hamiltonian(potential, mass, fraction)
            for fraction in zip(*bloch_fractions, strict=True)
        ]
    )
    operators = OperatorList(
        tuple_basis((list_basis, operators.basis[1])), operators.raw_data
    )

    return bloch_operator_from_list(operators)
