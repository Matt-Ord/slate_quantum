from __future__ import annotations

from itertools import starmap
from typing import TYPE_CHECKING, Any, Never, override

import numpy as np
from slate_core import Ctype, TupleBasis, TupleMetadata, basis
from slate_core import metadata as _metadata
from slate_core.basis import AsUpcast, BlockDiagonalBasis
from slate_core.metadata import (
    AxisDirections,
    LabeledMetadata,
    SpacedLengthMetadata,
    SpacedVolumeMetadata,
)

from slate_quantum import operator
from slate_quantum._util.legacy import (
    Metadata2D,
    StackedMetadata,
    tuple_basis,
)
from slate_quantum.bloch._shifted_basis import (
    BlochShiftedBasis,
)
from slate_quantum.bloch._transposed_basis import (
    BlochTransposedBasis,
    LegacyBlochTransposedBasis,
)
from slate_quantum.metadata import RepeatedLengthMetadata
from slate_quantum.operator._operator import (
    Operator,
    OperatorList,
    OperatorListBasis,
    OperatorMetadata,
)

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
    ) -> StackedMetadata[BlochFractionMetadata, None]:
        """Build a stacked metadata from a tuple of repeats."""
        return TupleMetadata(tuple(BlochFractionMetadata(n) for n in repeat), None)


def metadata_from_split(
    fraction_meta: StackedMetadata[BlochFractionMetadata, None],
    state_meta: SpacedVolumeMetadata,
) -> StackedMetadata[RepeatedLengthMetadata, AxisDirections]:
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
    metadata: StackedMetadata[RepeatedLengthMetadata, AxisDirections],
) -> LegacyBlochTransposedBasis[
    np.complexfloating,
    RepeatedLengthMetadata,
    AxisDirections,
]:
    """Build the Bloch basis."""
    operator_basis = basis.transformed_from_metadata(metadata)
    return BlochTransposedBasis(
        basis.with_modified_children(
            operator_basis, lambda _, i: BlochShiftedBasis(i).upcast()
        )
    ).upcast()


def _metadata_from_operator_list(
    meta: Metadata2D[
        StackedMetadata[BlochFractionMetadata, None],
        Metadata2D[SpacedVolumeMetadata, SpacedVolumeMetadata, None],
        None,
    ],
) -> LegacyBlochTransposedBasis[
    np.complexfloating,
    RepeatedLengthMetadata,
    AxisDirections,
]:
    """Get the metadata for the Bloch operator."""
    list_meta = meta.children[0]
    single_operator_meta = meta.children[1].children[0]
    full_operator_metadata = metadata_from_split(list_meta, single_operator_meta)
    return basis_from_metadata(full_operator_metadata)


def bloch_operator_from_list[
    M0: StackedMetadata[BlochFractionMetadata, None],
    M1: SpacedVolumeMetadata,
](
    operators: OperatorList[
        OperatorListBasis[M0, OperatorMetadata[M1]], np.dtype[np.complexfloating]
    ],
) -> Operator[
    AsUpcast[
        BlockDiagonalBasis[
            TupleBasis[
                tuple[
                    LegacyBlochTransposedBasis[
                        np.complexfloating, RepeatedLengthMetadata, AxisDirections
                    ],
                    LegacyBlochTransposedBasis[
                        np.complexfloating, RepeatedLengthMetadata, AxisDirections
                    ],
                ],
                None,
            ]
        ],
        OperatorMetadata[StackedMetadata[RepeatedLengthMetadata, AxisDirections]],
    ],
    np.dtype[np.complexfloating],
]:
    """Build the block diagonal Bloch Hamiltonian from a list of operators."""
    operators = operators.with_list_basis(
        basis.from_metadata(operators.basis.metadata().children[0])
    ).assert_ok()
    operators = operators.with_operator_basis(
        basis.transformed_from_metadata(
            operators.basis.inner.children[1].metadata(),
            is_dual=operators.basis.inner.children[1].is_dual,
        ).upcast()
    ).assert_ok()

    operator_basis = _metadata_from_operator_list(operators.basis.metadata())
    out_basis = BlockDiagonalBasis(
        tuple_basis((operator_basis, operator_basis.dual_basis())),
        operators.basis.metadata().children[1].shape,
    ).upcast()
    return Operator.build(out_basis, operators.raw_data).assert_ok()


def _get_sample_fractions[M: BlochFractionMetadata](
    metadata: StackedMetadata[M, None],
) -> tuple[np.ndarray[tuple[int], np.dtype[np.floating]], ...]:
    """Get the frequencies of the samples in a wavepacket, as a fraction of dk."""
    mesh = np.meshgrid(*(n.values for n in metadata.children), indexing="ij")
    return tuple(nki.ravel() for nki in mesh)


def kinetic_hamiltonian[M: SpacedLengthMetadata, E: AxisDirections](
    potential: Potential[M, E, Ctype[Never], np.dtype[np.complexfloating]],
    mass: float,
    repeat: tuple[int, ...],
) -> Operator[
    AsUpcast[
        BlockDiagonalBasis[
            TupleBasis[
                tuple[
                    LegacyBlochTransposedBasis[
                        np.complexfloating, RepeatedLengthMetadata, AxisDirections
                    ],
                    LegacyBlochTransposedBasis[
                        np.complexfloating, RepeatedLengthMetadata, AxisDirections
                    ],
                ],
                None,
            ]
        ],
        OperatorMetadata[StackedMetadata[RepeatedLengthMetadata, AxisDirections]],
    ],
    np.dtype[np.complexfloating],
]:
    """Build a kinetic Hamiltonian in the Bloch basis."""
    list_metadata = BlochFractionMetadata.from_repeats(repeat)
    bloch_fractions = _get_sample_fractions(list_metadata)
    list_basis = basis.from_metadata(list_metadata).upcast()

    operators = OperatorList.from_operators(
        [
            operator.build.kinetic_hamiltonian(potential, mass, np.array(fraction))
            for fraction in zip(*bloch_fractions, strict=True)
        ]
    )
    operators = OperatorList.build(
        tuple_basis((list_basis, operators.basis.inner.children[1])).upcast(),
        operators.raw_data,
    ).assert_ok()

    return bloch_operator_from_list(operators)
