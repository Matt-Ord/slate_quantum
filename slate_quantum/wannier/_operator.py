from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from surface_potential_analysis.basis.legacy import (
    FundamentalBasis,
    StackedBasisLike,
    StackedBasisWithVolumeLike,
    TupleBasisWithLengthLike,
)
from surface_potential_analysis.basis.util import (
    BasisUtil,
)
from surface_potential_analysis.operator.conversion import (
    convert_operator_to_basis,
)
from surface_potential_analysis.stacked_basis.conversion import (
    tuple_basis_as_fundamental,
)
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_to_basis,
)
from surface_potential_analysis.state_vector.eigenstate_calculation import (
    calculate_eigenvectors_hermitian,
    calculate_operator_inner_product,
)
from surface_potential_analysis.state_vector.state_vector import (
    LegacyStateVector,
    as_legacy_dual_vector,
)
from surface_potential_analysis.util.decorators import timed
from surface_potential_analysis.wavepacket.eigenstate_conversion import (
    unfurl_wavepacket,
)
from surface_potential_analysis.wavepacket.get_eigenstate import get_all_eigenstates
from surface_potential_analysis.wavepacket.wavepacket import (
    get_fundamental_unfurled_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.legacy import (
        BasisLike,
        BasisWithLengthLike,
        TupleBasisLike,
    )
    from surface_potential_analysis.operator.operator import SingleBasisOperator
    from surface_potential_analysis.wavepacket.wavepacket import (
        BlochWavefunctionListWithEigenvalues,
    )

    _SB0 = TypeVar("_SB0", bound=StackedBasisLike)

    _SBV0 = TypeVar("_SBV0", bound=StackedBasisWithVolumeLike)
    _SBV1 = TypeVar("_SBV1", bound=StackedBasisWithVolumeLike)

    _B0 = TypeVar("_B0", bound=BasisLike)
    _BL0 = TypeVar("_BL0", bound=BasisWithLengthLike)


def _get_position_operator(basis: _SBV0) -> SingleBasisOperator[_SBV0]:
    util = BasisUtil(basis)
    # We only get the location in the x0 direction here
    locations = util.fundamental_x_points_stacked[0]

    basis_position = tuple_basis_as_fundamental(basis)
    operator: SingleBasisOperator[Any] = {
        "basis": VariadicTupleBasis((basis_position, basis_position), None),
        "data": np.diag(locations),
    }
    return convert_operator_to_basis(operator, VariadicTupleBasis((basis, basis), None))


@timed
def _get_operator_between_states(
    states: list[LegacyStateVector[_SB0]], operator: SingleBasisOperator[_SB0]
) -> SingleBasisOperator[FundamentalBasis[Any]]:
    n_states = len(states)
    array = np.zeros((n_states, n_states), dtype=np.complex128)
    for i in range(n_states):
        dual_vector = as_legacy_dual_vector(states[i])
        for j in range(n_states):
            vector = states[j]
            array[i, j] = calculate_operator_inner_product(
                dual_vector, operator, vector
            )

    basis = FundamentalBasis(n_states)
    return {"data": array, "basis": VariadicTupleBasis((basis, basis), None)}


def _localize_operator(
    wavepacket: BlochWavefunctionListWithEigenvalues[_SB0, _SBV0],
    operator: SingleBasisOperator[_SBV1],
) -> list[BlochWavefunctionListWithEigenvalues[_SB0, _SBV0]]:
    states = [
        convert_state_vector_to_basis(state, operator["basis"][0])
        for state in get_all_eigenstates(wavepacket)
    ]
    operator_between_states = _get_operator_between_states(states, operator)
    eigenstates = calculate_eigenvectors_hermitian(operator_between_states)
    return [
        {
            "basis": wavepacket["basis"],
            "eigenvalue": wavepacket["eigenvalue"],
            "data": wavepacket["data"] * vector[:, np.newaxis],
        }
        for vector in eigenstates["data"]
    ]


def localize_position_operator(
    wavepacket: BlochWavefunctionListWithEigenvalues[_SB0, _SBV0],
) -> list[BlochWavefunctionListWithEigenvalues[_SB0, _SBV0]]:
    """
    Given a wavepacket generate a set of normalized wavepackets using the operator method.

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv, _B0Inv]

    Returns
    -------
    list[Wavepacket[_S0Inv, _B0Inv]]
    """
    basis = tuple_basis_as_fundamental(
        get_fundamental_unfurled_basis(wavepacket["basis"])
    )
    operator_position = _get_position_operator(basis)
    return _localize_operator(wavepacket, operator_position)


def localize_position_operator_many_band(
    wavepackets: list[
        BlochWavefunctionListWithEigenvalues[
            TupleBasisLike[*tuple[_B0, ...]],
            TupleBasisWithLengthLike[*tuple[_BL0, ...]],
        ]
    ],
) -> list[LegacyStateVector[Any]]:
    """
    Given a sequence of wavepackets at each band, get all possible eigenstates of position.

    Parameters
    ----------
    wavepackets : list[Wavepacket[_S0Inv, _B0Inv]]

    Returns
    -------
    list[StateVector[Any]]
    """
    basis = tuple_basis_as_fundamental(
        get_fundamental_unfurled_basis(wavepackets[0]["basis"])
    )
    states = [
        convert_state_vector_to_basis(state, basis)
        for wavepacket in wavepackets
        for state in get_all_eigenstates(wavepacket)
    ]
    operator_position = _get_position_operator(basis)
    operator = _get_operator_between_states(states, operator_position)
    eigenstates = calculate_eigenvectors_hermitian(operator)
    state_vectors = np.array([s["data"] for s in states], dtype=np.complex128)
    return [
        {
            "basis": basis,
            "data": np.tensordot(vector, state_vectors, axes=(0, 0)),
        }
        for vector in eigenstates["data"]
    ]


def localize_position_operator_many_band_individual(
    wavepackets: list[BlochWavefunctionListWithEigenvalues[_SB0, _SBV0]],
) -> list[LegacyStateVector[Any]]:
    """
    Given a wavepacket generate a set of normalized wavepackets using the operator method.

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv, _B0Inv]

    Returns
    -------
    list[Wavepacket[_S0Inv, _B0Inv]]
    """
    list_shape = (wavepackets[0]["basis"][0]).shape
    states = [
        unfurl_wavepacket(
            localize_position_operator(wavepacket)[np.prod(list_shape) // 4]
        )
        for wavepacket in wavepackets
    ]
    operator_position = _get_position_operator(states[0]["basis"])
    operator = _get_operator_between_states(states, operator_position)  # type: ignore[arg-type]
    eigenstates = calculate_eigenvectors_hermitian(operator)
    state_vectors = np.array([s["data"] for s in states])
    return [
        {
            "basis": states[0]["basis"],
            "data": np.tensordot(vector, state_vectors, axes=(0, 0)),
        }
        for vector in eigenstates["data"]
    ]
