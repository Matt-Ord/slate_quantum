from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from slate_core import (
    Array,
    Basis,
    BasisMetadata,
    SimpleMetadata,
    TupleBasis,
    TupleMetadata,
    basis,
)
from slate_core.basis import AsUpcast, DiagonalBasis

from slate_quantum.bloch._transposed_basis import BlochStateMetadata
from slate_quantum.bloch.build import StackedBlockedFractionMetadata
from slate_quantum.state._state import (
    StateList,
    StateListWithMetadata,
    StateWithMetadata,
)

if TYPE_CHECKING:
    from slate_core.metadata import SpacedVolumeMetadata

    from slate_quantum.operator._diagonal import DiagonalOperatorList
    from slate_quantum.operator._operator import (
        OperatorBasis,
        OperatorList,
        OperatorMetadata,
    )


# This is a super operator, which maps the bloch states
# into the wannier states.
#
# The bloch states are indexed by the band index and the bloch k index,
# we have n_band x n_bloch_k states, but each state is sparse - ie states at each bloch k
# only contain k points with the particular k states.
#
# The wannier states are indexed by the wannier index and the translation index.
# We have n_wannier x n_translation states, ie the same number of states as the bloch states.
# The difference is that the wannier states are dense - ie they are a sum of the contribution from each bloch k.
# The resulting state list is however sparse, since the wannier states can be recoverd
# from the 'fundamental' localized states by translating the states by a unit cell.
#
# This means that the localization operator is stored as
type LocalizationOperatorWithMetadata[
    MWannier: WannierListMetadata = WannierListMetadata,
    MBloch: BlochListMetadata = BlochListMetadata,
] = Array[
    Basis[LocalizationOperatorMetadata[MWannier, MBloch]],
    np.dtype[np.complexfloating],
]


type LocalizationOperatorMetadata[
    MWannier: WannierListMetadata = WannierListMetadata,
    MBloch: BlochListMetadata = BlochListMetadata,
] = TupleMetadata[
    tuple[
        # The lhs of the operator maps onto a wannier state list (wannier index, translation index) to state
        MWannier,
        # The rhs of the operator maps onto a bloch state list (band index, bloch k index) to state
        MBloch,
    ],
    None,
]

type LocalizationOperatorBasis[
    MWannier: WannierListMetadata = WannierListMetadata,
    MBloch: BlochListMetadata = BlochListMetadata,
] = TupleBasis[
    tuple[
        # The lhs of the operator maps onto a wannier state list (wannier index, translation index) to state
        Basis[MWannier],
        # The rhs of the operator maps onto a bloch state list (band index, bloch k index) to state
        Basis[MBloch],
    ],
    None,
]


"""
A list of operators, acting on each bloch k

List over the bloch k, each operator maps a series of
states at each band _B2 into the localized states made from
a mixture of each band _B1.

Note that the mixing between states of different bloch k
that is required to form the set of localized states is implicit.
The 'fundamental' localized states are a sum of the contribution from
each bloch k, and all other states can be found by translating the
states by a unit cell.
"""

# This is a localization operator, which only takes states from a single band
type DiagonalLocalizationOperator[
    MSingleRepeat: BasisMetadata,
    MBlochIndex: BasisMetadata,
] = LocalizationOperator[MSingleRepeat, MBlochIndex, MBlochIndex]


type WannierListMetadata[
    MWannier: SimpleMetadata = SimpleMetadata,
    MTranslation: BasisMetadata = BasisMetadata,
] = TupleMetadata[tuple[MWannier, MTranslation]]

type WannierStateListBasis[
    MWannier: WannierListMetadata = WannierListMetadata,
    MState: BlochStateMetadata = BlochStateMetadata,
    # TODO: a more precise type here
] = Basis[TupleMetadata[tuple[MWannier, MState], None]]
# The wannier states are indexed by the wannier index and the translation index.
# We have n_wannier x n_translation states, ie the same number of states as the bloch states.
# The difference is that the wannier states are dense - ie they are a sum of the contribution from each bloch k.
# The resulting state list is however sparse, since the wannier states can be recoverd
# from the 'fundamental' localized states by translating the states by a unit cell.
type WannierStateList[B: WannierStateListBasis, DT: np.dtype[np.complexfloating]] = (
    StateList[B, DT]
)

type WannierStateListWithMetadata[
    MWannier: WannierListMetadata = WannierListMetadata,
    MState: BlochStateMetadata = BlochStateMetadata,
    DT: np.dtype[np.complexfloating] = np.dtype[np.complexfloating],
] = WannierStateList[WannierStateListBasis[MWannier, MState], DT]


type BlochListMetadata[
    MBand: SimpleMetadata = SimpleMetadata,
    MBlochK: StackedBlockedFractionMetadata = StackedBlockedFractionMetadata,
] = TupleMetadata[tuple[MBand, MBlochK]]

type BlochStateListBasis[
    MBloch: BlochListMetadata = BlochListMetadata,
    MState: BlochStateMetadata = BlochStateMetadata,
    # TODO: a more precise type here
] = Basis[TupleMetadata[tuple[MBloch, MState], None]]
# The bloch states are indexed by the band index and the bloch k index,
# we have n_band x n_bloch_k states, but each state is sparse - ie states at each bloch k
# only contain k points with the particular k states.
type BlochStateList[B: BlochStateListBasis, DT: np.dtype[np.complexfloating]] = (
    StateList[B, DT]
)

type BlochStateListWithMetadata[
    MBloch: BlochListMetadata = BlochListMetadata,
    MState: BlochStateMetadata = BlochStateMetadata,
    DT: np.dtype[np.complexfloating] = np.dtype[np.complexfloating],
] = BlochStateList[BlochStateListBasis[MBloch, MState], DT]


def bloch_state_list_from_bloch_state[MState: BlochStateMetadata](
    state: StateWithMetadata[MState],
    # TODO: out metadata
) -> BlochStateListWithMetadata[BlochListMetadata, MState]:
    raise NotImplementedError


def state_list_from_bloch_state[MState: BlochStateMetadata](
    state: StateWithMetadata[MState],
    # TODO: out metadata
) -> StateListWithMetadata[StackedBlockedFractionMetadata, SpacedVolumeMetadata]:
    raise NotImplementedError


def localization_operator_as_diagonal[M0: BasisMetadata, M1: BasisMetadata](
    operator: LocalizationOperator[M0, M1, M1],
) -> DiagonalLocalizationOperator[M0, M1]:
    """Convert to a diagonal operator from full."""
    converted = convert_operator_list_to_basis(
        operator,
        TupleBasis((operator["basis"][1][0], operator["basis"][1][0]), None),
    )
    diagonal = operator_list_as_diagonal(converted)
    basis = TupleBasis((diagonal["basis"][0], diagonal["basis"][1][0]), None)
    return {"basis": TupleBasis((basis, basis), None), "data": diagonal["data"]}


def diagonal_localization_operator_as_full[M0: BasisMetadata, M1: BasisMetadata](
    operator: DiagonalLocalizationOperator[M0, M1],
) -> LocalizationOperator[M0, M1, M1]:
    """Convert to a full operator from diagonal."""
    return diagonal_operator_list_as_full(
        {
            "basis": TupleBasis(
                (
                    operator["basis"][0][0],
                    TupleBasis(
                        (operator["basis"][0][1], operator["basis"][0][1]), None
                    ),
                ),
                None,
            ),
            "data": operator["data"],
        }
    )


def localize_states[
    MWannier: WannierListMetadata,
    MBloch: BlochListMetadata,
    MState: BlochStateMetadata,
](
    wavepackets: BlochStateListWithMetadata[MBloch, MState],
    operator: LocalizationOperatorWithMetadata[MWannier, MBloch, MState],
) -> WannierStateListWithMetadata[MWannier, MState]:
    """
    Apply the LocalizationOperator to produce localized wavepackets.

    The localized states are indexed by the (wannier index, bloch k index) tuple.

    To recover the full wannier states, you need to combine each bloch k

    Parameters
    ----------
    wavepackets : WavepacketList[_B2, _SB1, _SB0]
        The unlocalized wavepackets
    operator : LocalizationOperator[_SB1, _B1, _B2]
        The operator used to localize the wavepackets

    Returns
    -------
    WavepacketList[_B1, _SB1, _SB0]
        The localized wavepackets
    """
    assert wavepackets["basis"][0][0] == operator["basis"][1][1]
    assert wavepackets["basis"][0][1] == operator["basis"][0]

    stacked_operator = operator["data"].reshape(
        operator["basis"][0].size, *operator["basis"][1].shape
    )
    vectors = wavepackets["data"].reshape(*wavepackets["basis"][0].shape, -1)

    # Sum over the bloch idx
    data = np.einsum("jil,ljk->ijk", stacked_operator, vectors)  # type:ignore lib

    return {
        "basis": TupleBasis(
            (
                TupleBasis((operator["basis"][1][0], wavepackets["basis"][0][1]), None),
                wavepackets["basis"][1],
            ),
            None,
        ),
        "data": data.reshape(-1),
    }


def get_localized_hamiltonian_from_eigenvalues[M0: BasisMetadata, M: BasisMetadata](
    # M0 here is per band - we should use some sort of sparse oeprator here instead...
    hamiltonian: DiagonalOperatorList[M0, _SB1],
    # M here is the localized basis, M0 is the bloch phase, and _SB1 is the bloch state basis
    operator: LocalizationOperator[_SB1, Basis[M], M0],
) -> OperatorList[
    AsUpcast[
        # M here is the localized basis, SB1 is the bais of a single repeat region
        TupleBasis[tuple[_SB1, OperatorBasis[M]], None],
        TupleMetadata[tuple[BasisMetadata, OperatorMetadata[M]], None],
    ]
]:
    """
    Localize the hamiltonian according to the Localization Operator.

    Parameters
    ----------
    hamiltonian : SingleBasisDiagonalOperatorList[_B2, _SB1]
    operator : LocalizationOperator[_SB1, _B1, _B2]

    Returns
    -------
    OperatorList[_SB1, _B1, _B1]
    """
    converted = np.einsum(  # type: ignore lib
        "dic,cd,djc->dij",
        operator["data"].reshape(-1, *operator["basis"][1].shape),
        hamiltonian["data"].reshape(
            hamiltonian["basis"][0].size, hamiltonian["basis"][1].shape[0]
        ),
        np.conj(operator["data"].reshape(-1, *operator["basis"][1].shape)),
    )
    return {
        "basis": TupleBasis(
            (
                hamiltonian["basis"][1][0],
                TupleBasis((operator["basis"][1][0], operator["basis"][1][0]), None),
            ),
            None,
        ),
        "data": converted.ravel(),
    }


def get_diagonal_localized_wavepackets(
    wavepackets: BlochWavefunctionListList[_B2, _SB1, _SB0],
    operator: DiagonalLocalizationOperator[_SB1, _B2],
) -> BlochWavefunctionListList[_B2, _SB1, _SB0]:
    """
    Apply the LocalizationOperator to produce localized wavepackets.

    Parameters
    ----------
    wavepackets : WavepacketList[_B2, _SB1, _SB0]
        The unlocalized wavepackets
    operator : LocalizationOperator[_SB1, _B1, _B2]
        The operator used to localize the wavepackets

    Returns
    -------
    WavepacketList[_B1, _SB1, _SB0]
        The localized wavepackets
    """
    assert wavepackets["basis"][0][0] == operator["basis"][0][1]
    assert wavepackets["basis"][0][1] == operator["basis"][0][0]

    stacked_operator = operator["data"].reshape(operator["basis"][0].shape)
    vectors = wavepackets["data"].reshape(*wavepackets["basis"][0].shape, -1)

    # Sum over the bloch idx
    data = np.einsum("ji,ijk->ijk", stacked_operator, vectors)  # type:ignore lib

    return {
        "basis": wavepackets["basis"],
        "data": data.reshape(-1),
    }


def get_identity_operator[M: LocalizationOperatorMetadata](
    metadata: M,
) -> Array[Basis[M], np.dtype[np.complexfloating]]:
    """
    Get the localization operator which is a simple identity.

    Parameters
    ----------
    basis : BlochWavefunctionListBasis[_SB0, _SB1]

    Returns
    -------
    LocalizationOperator[_SB1, FundamentalBasis[BasisMetadata], _SB0]
    """
    out_basis = AsUpcast(DiagonalBasis(basis.from_metadata(metadata)), metadata)
    return Array.build(
        out_basis,
        np.ones(out_basis.size, dtype=np.complex128),
    ).assert_ok()
