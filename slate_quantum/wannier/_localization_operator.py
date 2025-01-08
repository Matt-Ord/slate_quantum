from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import numpy as np
from slate.basis.stacked._tuple_basis import VariadicTupleBasis
from surface_potential_analysis.basis.legacy import (
    BasisLike,
    FundamentalBasis,
    StackedBasisLike,
    TupleBasis,
)
from surface_potential_analysis.operator.conversion import (
    convert_operator_list_to_basis,
)
from surface_potential_analysis.operator.operator import SingleBasisDiagonalOperator
from surface_potential_analysis.operator.operator_list import (
    OperatorList,
    SingleBasisDiagonalOperatorList,
    diagonal_operator_list_as_full,
    operator_list_as_diagonal,
)

if TYPE_CHECKING:
    from slate.metadata._metadata import BasisMetadata
    from surface_potential_analysis.wavepacket.wavepacket import (
        BlochWavefunctionListBasis,
        BlochWavefunctionListList,
    )

_B0 = TypeVar("_B0", bound=BasisLike)
_B1 = TypeVar("_B1", bound=BasisLike)
_B2 = TypeVar("_B2", bound=BasisLike)

LocalizationOperator = OperatorList[_B0, _B1, _B2]
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


DiagonalLocalizationOperator = SingleBasisDiagonalOperator[TupleBasis[_B0, _B1]]


def localization_operator_as_diagonal(
    operator: LocalizationOperator[_B0, _B1, _B2],
) -> DiagonalLocalizationOperator[_B0, _B1]:
    """Convert to a diagonal operator from full."""
    converted = convert_operator_list_to_basis(
        operator,
        VariadicTupleBasis((operator["basis"][1][0], operator["basis"][1][0]), None),
    )
    diagonal = operator_list_as_diagonal(converted)
    basis = VariadicTupleBasis((diagonal["basis"][0], diagonal["basis"][1][0]), None)
    return {"basis": VariadicTupleBasis((basis, basis), None), "data": diagonal["data"]}


def diagonal_localization_operator_as_full(
    operator: DiagonalLocalizationOperator[_B0, _B1],
) -> LocalizationOperator[_B0, _B1, _B1]:
    """Convert to a full operator from diagonal."""
    return diagonal_operator_list_as_full(
        {
            "basis": VariadicTupleBasis(
                (
                    operator["basis"][0][0],
                    VariadicTupleBasis(
                        (operator["basis"][0][1], operator["basis"][0][1]), None
                    ),
                ),
                None,
            ),
            "data": operator["data"],
        }
    )


_SB0 = TypeVar("_SB0", bound=StackedBasisLike)
_SB1 = TypeVar("_SB1", bound=StackedBasisLike)


def get_localized_wavepackets(
    wavepackets: BlochWavefunctionListList[_B2, _SB1, _SB0],
    operator: LocalizationOperator[_SB1, _B1, _B2],
) -> BlochWavefunctionListList[_B1, _SB1, _SB0]:
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
    assert wavepackets["basis"][0][0] == operator["basis"][1][1]
    assert wavepackets["basis"][0][1] == operator["basis"][0]

    stacked_operator = operator["data"].reshape(
        operator["basis"][0].size, *operator["basis"][1].shape
    )
    vectors = wavepackets["data"].reshape(*wavepackets["basis"][0].shape, -1)

    # Sum over the bloch idx
    data = np.einsum("jil,ljk->ijk", stacked_operator, vectors)  # type:ignore lib

    return {
        "basis": VariadicTupleBasis(
            (
                VariadicTupleBasis(
                    (operator["basis"][1][0], wavepackets["basis"][0][1]), None
                ),
                wavepackets["basis"][1],
            ),
            None,
        ),
        "data": data.reshape(-1),
    }


def get_localized_hamiltonian_from_eigenvalues(
    hamiltonian: SingleBasisDiagonalOperatorList[_B2, _SB1],
    operator: LocalizationOperator[_SB1, _B1, _B2],
) -> OperatorList[_SB1, _B1, _B1]:
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
        "basis": VariadicTupleBasis(
            (
                hamiltonian["basis"][1][0],
                VariadicTupleBasis(
                    (operator["basis"][1][0], operator["basis"][1][0]), None
                ),
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


def get_identity_operator(
    basis: BlochWavefunctionListBasis[_SB0, _SB1],
) -> LocalizationOperator[_SB1, FundamentalBasis[BasisMetadata], _SB0]:
    """
    Get the localization operator which is a simple identity.

    Parameters
    ----------
    basis : BlochWavefunctionListBasis[_SB0, _SB1]

    Returns
    -------
    LocalizationOperator[_SB1, FundamentalBasis[BasisMetadata], _SB0]
    """
    return diagonal_operator_list_as_full(
        {
            "basis": VariadicTupleBasis(
                (
                    basis[1],
                    VariadicTupleBasis((FundamentalBasis(basis[1].n), None), basis[0]),
                ),
                None,
            ),
            "data": np.ones(basis.n, dtype=np.complex128),
        }
    )
