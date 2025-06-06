from __future__ import annotations

from itertools import starmap
from typing import TYPE_CHECKING, Any, Never, override

import numpy as np
from slate_core import Ctype, TupleBasis, TupleMetadata, basis
from slate_core import metadata as _metadata
from slate_core.basis import BlockDiagonalBasis
from slate_core.metadata import (
    AxisDirections,
    EvenlySpacedLengthMetadata,
    EvenlySpacedVolumeMetadata,
    LabeledMetadata,
)

from slate_quantum import operator
from slate_quantum.bloch._shifted_basis import (
    BlochShiftedBasis,
)
from slate_quantum.bloch._transposed_basis import (
    BlochOperatorBasis,
    BlochStateBasis,
    BlochStateMetadata,
    BlochTransposedBasis,
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
        """The individual bloch fractions."""
        return _metadata.fundamental_nk_points(self) / self.fundamental_size

    @staticmethod
    def from_repeats(
        repeat: tuple[int, ...],
    ) -> StackedBlockedFractionMetadata:
        """Build a stacked metadata from a tuple of repeats."""
        return TupleMetadata(tuple(BlochFractionMetadata(n) for n in repeat), None)


type StackedBlockedFractionMetadata = TupleMetadata[
    tuple[BlochFractionMetadata, ...], None
]


def bloch_state_metadata_from_split(
    fraction_meta: StackedBlockedFractionMetadata,
    state_meta: EvenlySpacedVolumeMetadata,
) -> BlochStateMetadata:
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


def fraction_metadata_from_bloch_state(
    metadata: BlochStateMetadata,
) -> StackedBlockedFractionMetadata:
    """Get the metadata for the Bloch operator."""
    return BlochFractionMetadata.from_repeats(
        tuple(c.n_repeats for c in metadata.children)
    )


def basis_from_metadata(
    metadata: BlochStateMetadata,
) -> BlochStateBasis:
    """Build the Bloch basis."""
    operator_basis = basis.transformed_from_metadata(metadata)
    return BlochTransposedBasis(
        basis.with_modified_children(
            operator_basis, lambda _, i: BlochShiftedBasis(i).upcast()
        )
    ).upcast()


def _metadata_from_operator_list(
    meta: TupleMetadata[
        tuple[
            TupleMetadata[tuple[BlochFractionMetadata, ...], None],
            TupleMetadata[
                tuple[EvenlySpacedVolumeMetadata, EvenlySpacedVolumeMetadata], None
            ],
        ],
        None,
    ],
) -> BlochStateBasis:
    """Get the metadata for the Bloch operator."""
    list_meta = meta.children[0]
    single_operator_meta = meta.children[1].children[0]
    full_operator_metadata = bloch_state_metadata_from_split(
        list_meta, single_operator_meta
    )
    return basis_from_metadata(full_operator_metadata)


def bloch_operator_from_list[
    M0: TupleMetadata[tuple[BlochFractionMetadata, ...], None],
    M1: EvenlySpacedVolumeMetadata,
](
    operators: OperatorList[
        OperatorListBasis[M0, OperatorMetadata[M1]], np.dtype[np.complexfloating]
    ],
) -> Operator[BlochOperatorBasis, np.dtype[np.complexfloating]]:
    """Build the block diagonal Bloch Hamiltonian from a list of operators."""
    operators = operators.with_list_basis(
        basis.from_metadata(operators.basis.metadata().children[0])
    )
    operators = operators.with_operator_basis(
        basis.transformed_from_metadata(
            operators.basis.metadata().children[1],
            is_dual=operators.basis.inner.is_dual[1],
        ).upcast()
    )

    operator_basis = _metadata_from_operator_list(operators.basis.metadata())
    out_basis = BlockDiagonalBasis(
        TupleBasis((operator_basis, operator_basis.dual_basis())),
        operators.basis.metadata().children[1].shape,
    ).upcast()
    return Operator(out_basis, operators.raw_data)


def get_sample_fractions(
    metadata: StackedBlockedFractionMetadata,
) -> tuple[np.ndarray[tuple[int], np.dtype[np.floating]], ...]:
    """Get the frequencies of the samples in a wavepacket, as a fraction of dk."""
    mesh = np.meshgrid(*(n.values for n in metadata.children), indexing="ij")
    return tuple(nki.ravel() for nki in mesh)


def kinetic_hamiltonian[M: EvenlySpacedLengthMetadata, E: AxisDirections](
    potential: Potential[M, E, Ctype[Never], np.dtype[np.complexfloating]],
    mass: float,
    repeat: tuple[int, ...],
) -> Operator[BlochOperatorBasis, np.dtype[np.complexfloating]]:
    """Build a kinetic Hamiltonian in the Bloch basis."""
    list_metadata = BlochFractionMetadata.from_repeats(repeat)
    bloch_fractions = get_sample_fractions(list_metadata)
    list_basis = basis.from_metadata(list_metadata).upcast()

    operators = OperatorList.from_operators(
        [
            operator.build.kinetic_hamiltonian(potential, mass, np.array(fraction))
            for fraction in zip(*bloch_fractions, strict=True)
        ]
    )
    operators = OperatorList(
        TupleBasis((list_basis, operators.basis.inner.children[1])).upcast(),
        operators.raw_data,
    )

    return bloch_operator_from_list(operators)
