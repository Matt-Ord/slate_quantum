from __future__ import annotations

from itertools import starmap
from typing import TYPE_CHECKING, Any, Never, override

import numpy as np
from slate_core import Basis, Ctype, TupleBasis, TupleBasisLike, TupleMetadata, basis
from slate_core import metadata as _metadata
from slate_core.basis import AsUpcast, BlockDiagonalBasis
from slate_core.metadata import (
    AxisDirections,
    LabeledMetadata,
    SpacedLengthMetadata,
    SpacedVolumeMetadata,
)

from slate_quantum import operator
from slate_quantum.bloch._shifted_basis import BlochShiftedBasis
from slate_quantum.bloch._transposed_basis import BlochTransposedBasis
from slate_quantum.metadata import RepeatedLengthMetadata
from slate_quantum.operator._operator import Operator, OperatorList, OperatorMetadata

if TYPE_CHECKING:
    from slate_quantum.operator._diagonal import Potential


class BlochFractionMetadata(LabeledMetadata[np.dtype[np.floating]]):
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
    ) -> TupleMetadata[tuple[BlochFractionMetadata, ...], None]:
        """Build a stacked metadata from a tuple of repeats."""
        return TupleMetadata(tuple(BlochFractionMetadata(n) for n in repeat), None)


def metadata_from_split(
    fraction_meta: TupleMetadata[tuple[BlochFractionMetadata, ...], None],
    state_meta: SpacedVolumeMetadata,
) -> TupleMetadata[tuple[RepeatedLengthMetadata, ...], AxisDirections]:
    """Get the metadata for the Bloch operator."""
    return TupleMetadata(
        tuple(
            starmap(
                RepeatedLengthMetadata,
                zip(state_meta.children, fraction_meta.shape, strict=True),
            )
        ),
        state_meta.extra,
    )


def basis_from_metadata(
    metadata: TupleMetadata[tuple[RepeatedLengthMetadata, ...], AxisDirections],
) -> BlochTransposedBasis[
    TupleBasis[tuple[Basis[RepeatedLengthMetadata], ...], AxisDirections],
    Ctype[np.complexfloating],
]:
    """Build the Bloch basis."""
    operator_basis = basis.transformed_from_metadata(metadata)
    return BlochTransposedBasis(
        basis.with_modified_children(
            operator_basis, lambda _, i: BlochShiftedBasis(i).resolve_ctype().upcast()
        )
    ).resolve_ctype()


def _metadata_from_operator_list(
    meta: TupleMetadata[
        tuple[
            TupleMetadata[tuple[BlochFractionMetadata, ...], None],
            TupleMetadata[tuple[SpacedVolumeMetadata, SpacedVolumeMetadata], None],
        ],
        None,
    ],
) -> BlochTransposedBasis[
    TupleBasis[tuple[Basis[RepeatedLengthMetadata], ...], AxisDirections],
    Ctype[np.complexfloating],
]:
    """Get the metadata for the Bloch operator."""
    list_meta = meta.children[0]
    single_operator_meta = meta.children[1].children[0]
    full_operator_metadata = metadata_from_split(list_meta, single_operator_meta)
    return basis_from_metadata(full_operator_metadata)


type TransposedRepeatBasis[CT: Ctype[Never] = Ctype[Never]] = BlochTransposedBasis[
    TupleBasis[tuple[Basis[RepeatedLengthMetadata], ...], AxisDirections],
    Ctype[np.complexfloating],
]
type BlochOperatorBasis[CT: Ctype[Never] = Ctype[Never]] = BlockDiagonalBasis[
    TupleBasis[
        tuple[TransposedRepeatBasis[CT], TransposedRepeatBasis[CT]],
        None,
        CT,
    ],
    CT,
]


def bloch_operator_from_list[
    M0: TupleMetadata[tuple[BlochFractionMetadata, ...], None],
    M1: SpacedVolumeMetadata,
](
    operators: OperatorList[
        TupleBasisLike[tuple[M0, OperatorMetadata[M1]]], np.dtype[np.complexfloating]
    ],
) -> Operator[
    AsUpcast[BlochOperatorBasis, OperatorMetadata], np.dtype[np.complexfloating]
]:
    """Build the block diagonal Bloch Hamiltonian from a list of operators."""
    operators_as_list = operators.with_list_basis(
        basis.from_metadata(operators.basis.metadata().children[0])
    ).assert_ok()
    operators_transformed = operators_as_list.with_operator_basis(
        basis.transformed_from_metadata(
            operators_as_list.basis.inner.children[1].metadata(),
            is_dual=operators_as_list.basis.inner.children[1].is_dual,
        ).upcast()
    ).assert_ok()

    operator_basis = _metadata_from_operator_list(operators.basis.metadata())
    out_basis = BlockDiagonalBasis(
        TupleBasis((operator_basis, operator_basis.dual_basis())),
        operators_transformed.basis.metadata().children[1].shape,
    ).upcast()
    return Operator.build(out_basis, operators.raw_data).assert_ok()


def _get_sample_fractions[M: BlochFractionMetadata](
    metadata: TupleMetadata[tuple[M, ...], None],
) -> tuple[np.ndarray[tuple[int], np.dtype[np.floating]], ...]:
    """Get the frequencies of the samples in a wavepacket, as a fraction of dk."""
    mesh = np.meshgrid(*(n.values for n in metadata.children), indexing="ij")
    return tuple(nki.ravel() for nki in mesh)


def kinetic_hamiltonian[M: SpacedLengthMetadata, E: AxisDirections](
    potential: Potential[M, E, Ctype[np.complexfloating], np.dtype[np.complexfloating]],
    mass: float,
    repeat: tuple[int, ...],
) -> Operator[
    AsUpcast[BlochOperatorBasis, OperatorMetadata],
    np.dtype[np.complexfloating],
]:
    """Build a kinetic Hamiltonian in the Bloch basis."""
    list_metadata = BlochFractionMetadata.from_repeats(repeat)
    bloch_fractions = _get_sample_fractions(list_metadata)
    basis.from_metadata(list_metadata)

    operators = OperatorList.from_operators(
        [
            operator.build.kinetic_hamiltonian(potential, mass, np.array(fraction))
            for fraction in zip(*bloch_fractions, strict=True)
        ]
    )
    cast_operators = OperatorList.build(
        TupleBasis(
            (basis.from_metadata(list_metadata), operators.basis.inner.children[1])
        ).upcast(),
        operators.raw_data,
    ).assert_ok()

    return bloch_operator_from_list(cast_operators)
