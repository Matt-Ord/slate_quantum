from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import scipy.linalg  # type:ignore lib

from surface_potential_analysis.basis.legacy import (
    BasisLike,
    FundamentalBasis,
    StackedBasisLike,
    StackedBasisWithVolumeLike,
    TupleBasis,
    TupleBasisWithLengthLike,
)
from surface_potential_analysis.basis.util import (
    BasisUtil,
)
from surface_potential_analysis.stacked_basis.conversion import (
    tuple_basis_as_fundamental,
)
from surface_potential_analysis.stacked_basis.util import wrap_index_around_origin
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_list_to_basis,
    convert_state_vector_to_momentum_basis,
    convert_state_vector_to_position_basis,
)
from surface_potential_analysis.state_vector.state_vector_list import (
    as_state_vector_list,
    get_state_vector,
)
from surface_potential_analysis.state_vector.state_vector_list import (
    calculate_inner_products as calculate_inner_product_states,
)
from surface_potential_analysis.state_vector.util import (
    get_single_point_state_vector_excact,
)
from surface_potential_analysis.wavepacket.get_eigenstate import (
    get_states_at_bloch_idx,
    get_tight_binding_state,
    get_tight_binding_states,
    get_wavepacket_state_vector,
)
from surface_potential_analysis.wavepacket.localization_operator import (
    LocalizationOperator,
    get_localized_wavepackets,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    BlochWavefunctionListBasis,
    BlochWavefunctionListList,
    as_wavepacket_list,
    get_fundamental_unfurled_basis,
    get_wavepacket,
)

if TYPE_CHECKING:
    from slate.metadata._metadata import BasisMetadata

    from surface_potential_analysis.basis.legacy import (
        FundamentalPositionBasis,
    )
    from surface_potential_analysis.operator.operator import LegacyOperator
    from surface_potential_analysis.state_vector.state_vector import (
        LegacyStateVector,
    )
    from surface_potential_analysis.state_vector.state_vector_list import (
        LegacyStateVectorList,
    )
    from surface_potential_analysis.types import (
        SingleIndexLike,
        SingleStackedIndexLike,
    )
    from surface_potential_analysis.wavepacket.wavepacket import (
        BlochWavefunctionList,
    )

_SB0 = TypeVar("_SB0", bound=StackedBasisLike)

_SBV0 = TypeVar("_SBV0", bound=StackedBasisWithVolumeLike)
_SBV1 = TypeVar("_SBV1", bound=StackedBasisWithVolumeLike)

_B0 = TypeVar("_B0", bound=BasisLike)
_B1 = TypeVar("_B1", bound=BasisLike)
_B2 = TypeVar("_B2", bound=BasisLike)
_B3 = TypeVar("_B3", bound=BasisLike)


def get_state_projections_many_band(
    states: LegacyStateVectorList[_B0, _B2],
    projections: LegacyStateVectorList[_B1, _B3],
) -> LegacyOperator[_B0, _B1]:
    converted = convert_state_vector_list_to_basis(projections, states["basis"][1])
    return calculate_inner_product_states(states, converted)


def _get_orthogonal_projected_states_many_band(
    states: LegacyStateVectorList[_B0, _SBV0],
    projections: LegacyStateVectorList[_B1, _B2],
) -> LegacyOperator[_B1, _B0]:
    projected = get_state_projections_many_band(states, projections)
    # Use SVD to generate orthogonal matrix u v_dagger
    u, _s, v_dagger = scipy.linalg.svd(  # type:ignore lib
        projected["data"].reshape(projected["basis"].shape),
        full_matrices=False,
        compute_uv=True,
        overwrite_a=False,
        check_finite=True,
    )
    orthonormal_a = np.tensordot(u, v_dagger, axes=(1, 0))  # type:ignore lib
    return {
        "basis": VariadicTupleBasis(
            (projections["basis"][0], states["basis"][0]), None
        ),
        "data": orthonormal_a.T.reshape(-1),  # Maybe this should be .conj().T ??
    }


def get_localization_operator_for_projections(
    wavepackets: BlochWavefunctionListList[_B0, _SB0, _SBV0],
    projections: LegacyStateVectorList[_B1, _B2],
) -> LocalizationOperator[_SB0, _B1, _B0]:
    converted = convert_state_vector_list_to_basis(
        wavepackets,
        stacked_basis_as_transformed_basis(wavepackets["basis"][1]),
    )
    # Note here we localize each bloch k seperately
    states = [
        get_states_at_bloch_idx(converted, idx)  # type: ignore can't ensure WavepacketList is a stacked fundamental basis, and still have the right return type
        for idx in range(converted["basis"][0][1].n)
    ]
    data = [
        _get_orthogonal_projected_states_many_band(s, projections)["data"]
        for s in states
    ]
    return {
        "basis": TupleBasis(
            wavepackets["basis"][0][1],
            VariadicTupleBasis(
                (projections["basis"][0], wavepackets["basis"][0][0]), None
            ),
        ),
        "data": np.array(data, dtype=np.complex128).reshape(-1),
    }


def localize_wavepacket_projection(
    wavepackets: BlochWavefunctionListList[_B0, _SB0, _SBV0],
    projections: LegacyStateVectorList[_B1, _B2],
) -> BlochWavefunctionListList[_B1, _SB0, _SBV0]:
    """
    Given a wavepacket, localize using the given projection.

    Parameters
    ----------
    wavepacket : Wavepacket[_B0Inv, _B0Inv]
    projection : StateVector[_B1Inv]

    Returns
    -------
    Wavepacket[_B0Inv, _B0Inv]
    """
    operator = get_localization_operator_for_projections(wavepackets, projections)
    return get_localized_wavepackets(wavepackets, operator)


def localize_single_band_wavepacket_projection(
    wavepacket: BlochWavefunctionList[_SB0, _SBV0],
    projection: LegacyStateVector[_SBV1],
) -> BlochWavefunctionList[_SB0, _SBV0]:
    """
    Given a wavepacket, localize using the given projection.

    Parameters
    ----------
    wavepacket : Wavepacket[_B0Inv, _B0Inv]
    projection : StateVector[_B1Inv]

    Returns
    -------
    Wavepacket[_B0Inv, _B0Inv]
    """
    wavepackets = as_wavepacket_list([wavepacket])
    projections = as_state_vector_list([projection])

    return get_wavepacket(localize_wavepacket_projection(wavepackets, projections), 0)


def get_localization_operator_tight_binding_projections(
    wavepackets: BlochWavefunctionListList[_B0, _SB0, _SBV0],
) -> LocalizationOperator[_SB0, _B0, _B0]:
    projections = get_tight_binding_states(wavepackets)
    # Better performace if we provide the projection in transformed basis
    converted = convert_state_vector_list_to_basis(
        projections,
        stacked_basis_as_transformed_basis(projections["basis"][1]),
    )
    return get_localization_operator_for_projections(wavepackets, converted)


def localize_tight_binding_projection(
    wavepacket: BlochWavefunctionList[_SB0, _SBV0],
) -> BlochWavefunctionList[_SB0, _SBV0]:
    """
    Given a wavepacket, localize using a tight binding projection.

    Parameters
    ----------
    wavepacket : Wavepacket[_B0Inv, _B0Inv]

    Returns
    -------
    Wavepacket[_B0Inv, _B0Inv]
    """
    # Initial guess is that the localized state is just the state of some eigenstate
    # truncated at the edge of the first
    # unit cell, centered at the two point max of the wavefunction
    projection = get_tight_binding_state(wavepacket)
    # Better performace if we provide the projection in transformed basis
    transformed = convert_state_vector_to_momentum_basis(projection)
    return localize_single_band_wavepacket_projection(wavepacket, transformed)


def get_single_point_state_for_wavepacket(
    wavepacket: BlochWavefunctionList[_SB0, _SBV0],
    idx: SingleIndexLike = 0,
    origin: SingleStackedIndexLike | None = None,
) -> LegacyStateVector[TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]]]:
    state_0 = convert_state_vector_to_position_basis(
        get_wavepacket_state_vector(wavepacket, idx)
    )
    util = BasisUtil(state_0["basis"])
    if origin is None:
        idx_0: SingleStackedIndexLike = util.get_stacked_index(
            int(np.argmax(np.abs(state_0["data"]), axis=-1))
        )
        origin = wrap_index_around_origin(state_0["basis"], idx_0)
    return get_single_point_state_vector_excact(
        state_0["basis"], util.get_flat_index(origin, mode="wrap")
    )


def localize_single_point_projection(
    wavepacket: BlochWavefunctionList[_SB0, _SBV0],
    idx: SingleIndexLike = 0,
) -> BlochWavefunctionList[_SB0, _SBV0]:
    """
    Given a wavepacket, localize using a tight binding projection.

    Parameters
    ----------
    wavepacket : Wavepacket[_B0Inv, _B0Inv]

    Returns
    -------
    Wavepacket[_B0Inv, _B0Inv]
    """
    # Initial guess is that the localized state is just the state of some eigenstate
    # truncated at the edge of the first
    # unit cell, centered at the two point max of the wavefunction
    projection = get_single_point_state_for_wavepacket(wavepacket, idx)
    # Will have better performace if we provide it in a truncated position basis
    return localize_single_band_wavepacket_projection(wavepacket, projection)


def get_exponential_state(
    wavepacket: BlochWavefunctionList[_SB0, _SBV0],
    idx: SingleIndexLike = 0,
    origin: SingleIndexLike | None = None,
) -> LegacyStateVector[TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]]]:
    """
    Given a wavepacket, get the state decaying exponentially from the maximum.

    Parameters
    ----------
    wavepacket : Wavepacket[_B0Inv, _B0Inv]
        The initial wavepacket
    idx : SingleIndexLike, optional
        The index of the state vector to use as reference, by default 0
    origin : SingleIndexLike | None, optional
        The origin about which to produce the localized state, by default the maximum of the wavefunction

    Returns
    -------
    StateVector[tuple[FundamentalPositionBasis, ...]]
        The localized state under the tight binding approximation
    """
    state_0 = convert_state_vector_to_position_basis(
        get_wavepacket_state_vector(wavepacket, idx)
    )

    util = BasisUtil(state_0["basis"])
    origin = (
        util.get_stacked_index(int(np.argmax(np.abs(state_0["data"]), axis=-1)))
        if origin is None
        else origin
    )
    origin_stacked = (
        origin if isinstance(origin, tuple) else util.get_stacked_index(origin)
    )
    origin_stacked = wrap_index_around_origin(wavepacket["basis"], origin_stacked)

    coordinates = wrap_index_around_origin(
        state_0["basis"], util.stacked_nx_points, origin=origin_stacked
    )
    unit_cell_util = BasisUtil(wavepacket["basis"])
    dx0 = coordinates[0] - origin_stacked[0] / unit_cell_util.fundamental_shape[0]
    dx1 = coordinates[1] - origin_stacked[1] / unit_cell_util.fundamental_shape[1]
    dx2 = coordinates[2] - origin_stacked[2] / unit_cell_util.fundamental_shape[2]

    out: LegacyStateVector[
        TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]]
    ] = {
        "basis": state_0["basis"],
        "data": np.zeros_like(state_0["data"]),
    }
    out["data"] = np.exp(-(dx0**2 + dx1**2 + dx2**2))
    out["data"] /= np.linalg.norm(out["data"])  # type: ignore can be float
    return out


def _get_exponential_decay_state(
    wavepacket: BlochWavefunctionList[_SB0, _SBV0],
) -> LegacyStateVector[TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]]]:
    exponential = get_exponential_state(wavepacket)
    tight_binding = convert_state_vector_to_position_basis(
        get_wavepacket_state_vector(wavepacket, 0)
    )
    out: LegacyStateVector[
        TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]]
    ] = {
        "basis": exponential["basis"],
        "data": exponential["data"] * tight_binding["data"],
    }
    out["data"] /= np.linalg.norm(out["data"])
    return out


def localize_exponential_decay_projection(
    wavepacket: BlochWavefunctionList[_SB0, _SBV0],
) -> BlochWavefunctionList[_SB0, _SBV0]:
    """
    Given a wavepacket, localize using a tight binding projection.

    Parameters
    ----------
    wavepacket : Wavepacket[_B0Inv, _B0Inv]

    Returns
    -------
    Wavepacket[_B0Inv, _B0Inv]
    """
    # Initial guess is that the localized state is the tight binding state
    # multiplied by an exponential
    projection = _get_exponential_decay_state(wavepacket)
    return localize_single_band_wavepacket_projection(wavepacket, projection)


def get_gaussian_states(
    wavepacket: BlochWavefunctionList[_SB0, _SBV0],
    origin: SingleIndexLike = 0,
) -> LegacyStateVectorList[
    FundamentalBasis[BasisMetadata],
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],
]:
    """
    Given a wavepacket, get the state decaying exponentially from the maximum.

    Parameters
    ----------
    wavepacket : Wavepacket[_B0Inv, _B0Inv]
        The initial wavepacket
    idx : SingleIndexLike, optional
        The index of the state vector to use as reference, by default 0
    origin : SingleIndexLike | None, optional
        The origin about which to produce the localized state, by default the maximum of the wavefunction

    Returns
    -------
    StateVector[tuple[FundamentalPositionBasis, ...]]
        The localized state under the tight binding approximation
    """
    basis = tuple_basis_as_fundamental(
        get_fundamental_unfurled_basis(wavepacket["basis"])
    )
    util = BasisUtil(basis)
    origin_stacked = (
        origin if isinstance(origin, tuple) else util.get_stacked_index(origin)
    )
    origin_stacked = wrap_index_around_origin(wavepacket["basis"], origin_stacked)

    coordinates = wrap_index_around_origin(
        basis, util.stacked_nx_points, origin=origin_stacked
    )
    unit_cell_shape = (wavepacket["basis"]).shape
    dx = tuple(
        (c - o) / w
        for (c, o, w) in zip(coordinates, origin_stacked, unit_cell_shape, strict=True)
    )

    out: LegacyStateVectorList[
        FundamentalBasis[BasisMetadata],
        TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],
    ] = {
        "basis": VariadicTupleBasis((FundamentalBasis(1), None), basis),
        "data": np.zeros(basis.n, dtype=np.complex128),
    }
    out["data"] = np.exp(-0.5 * np.sum(np.square(dx), axis=(0)))
    out["data"] /= np.linalg.norm(out["data"])
    return out


def localize_wavepacket_gaussian_projection(
    wavepacket: BlochWavefunctionList[_SB0, _SBV0],
) -> BlochWavefunctionList[_SB0, _SBV0]:
    """
    Given a wavepacket, localize using a tight binding projection.

    Parameters
    ----------
    wavepacket : Wavepacket[_B0Inv, _B0Inv]

    Returns
    -------
    Wavepacket[_B0Inv, _B0Inv]
    """
    # Initial guess is that the localized state is the tight binding state
    # multiplied by an exponential
    projection = get_state_vector(get_gaussian_states(wavepacket), 0)
    # Better performace if we provide the projection in transformed basis
    projection = convert_state_vector_to_momentum_basis(projection)
    return localize_single_band_wavepacket_projection(wavepacket, projection)


def get_evenly_spaced_points(
    basis: BlochWavefunctionListBasis[Any, Any], shape: tuple[int, ...]
) -> LegacyStateVectorList[
    TupleBasis[*tuple[FundamentalBasis[BasisMetadata], ...]],
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],
]:
    fundamental_basis = tuple_basis_as_fundamental(
        get_fundamental_unfurled_basis(basis)
    )
    util = BasisUtil(fundamental_basis)

    out = np.zeros((np.prod(shape), fundamental_basis.n), dtype=np.complex128)

    for i, idx in enumerate(np.ndindex(shape)):
        sample_point = tuple(
            (n * idx_i) // s
            for (n, idx_i, s) in zip(util.shape, idx, shape, strict=True)
        )
        out[i, util.get_flat_index(sample_point)] = 1

    return {
        "basis": TupleBasis(
            TupleBasis(*tuple(FundamentalBasis(s) for s in shape)),
            fundamental_basis,
        ),
        "data": out.ravel(),
    }
