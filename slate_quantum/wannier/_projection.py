from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.linalg  # type:ignore lib
from slate_core import Array, Basis, BasisMetadata, TupleBasis, TupleMetadata, array
from slate_core.metadata import SpacedVolumeMetadata

from slate_quantum.bloch._transposed_basis import BlochStateMetadata
from slate_quantum.state._state import (
    StateList,
    StateWithMetadata,
    all_inner_product,
)
from slate_quantum.wannier._localization_operator import (
    BlochListMetadata,
    BlochStateListWithMetadata,
    LocalizationOperatorWithMetadata,
    WannierListMetadata,
    WannierStateListWithMetadata,
    bloch_state_list_from_state,
    localize_states,
)

if TYPE_CHECKING:
    from slate_quantum.state import StateListWithMetadata


def _get_orthogonal_projected_states_many_band[
    M0: BasisMetadata,
    M1: SpacedVolumeMetadata,
    M2: BasisMetadata,
    M3: BasisMetadata,
](
    states: StateListWithMetadata[M0, M1],
    projections: StateListWithMetadata[M2, M3],
) -> Array[TupleBasis[tuple[Basis[M0], Basis[M2]], None], np.dtype[np.floating]]:
    projected = array.as_tuple_basis(all_inner_product(states, projections))
    # Use SVD to generate orthogonal matrix u v_dagger
    u, _s, v_dagger = scipy.linalg.svd(  # type:ignore lib
        projected.raw_data.reshape(projected.basis.shape),
        full_matrices=False,
        compute_uv=True,
        overwrite_a=False,
        check_finite=True,
    )
    orthonormal_a = np.tensordot(u, v_dagger, axes=(1, 0))  # type:ignore lib
    return Array.build(projected.basis, orthonormal_a.T).assert_ok()


def get_localization_operator_for_projections[
    MBloch: BlochListMetadata,
    MState: BlochStateMetadata,
](
    states: BlochStateListWithMetadata[MBloch, MState],
    projections: StateList[Basis[TupleMetadata[tuple[BasisMetadata, MState], Any]]],
    # TODO: what WannierListMetadata metadata here?
) -> LocalizationOperatorWithMetadata[WannierListMetadata, MBloch]:
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
        _get_orthogonal_projected_states_many_band(s, projections).raw_data
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


def localize_states_projection[
    MBloch: BlochListMetadata,
    MState: BlochStateMetadata,
](
    states: BlochStateListWithMetadata[MBloch, MState],
    projections: StateList[Basis[TupleMetadata[tuple[BasisMetadata, MState], Any]]],
    # TODO: what out metadata here?
) -> WannierStateListWithMetadata[WannierListMetadata, MState]:
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
    operator = get_localization_operator_for_projections(states, projections)
    return localize_states(states, operator)


def localize_state_projection[
    MState: BlochStateMetadata,
](
    state: StateWithMetadata[MState],
    projection: StateWithMetadata[MState],
    # TODO: out Metadata here?
) -> StateWithMetadata[MState]:
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
    projections = StateList.from_states([projection])
    states = bloch_state_list_from_state(state)
    return localize_states_projection(states, projections)[0, :]


def get_tight_binding_states[
    MBloch: BlochListMetadata,
    MState: BlochStateMetadata,
](
    states: BlochStateListWithMetadata[MBloch, MState],
) -> StateList[Basis[TupleMetadata[tuple[BasisMetadata, MState], Any]]]:
    # Initial guess is that the localized state is just the state of some eigenstate
    # truncated at the edge of the first
    # unit cell, centered at the two point max of the wavefunction
    raise NotImplementedError


def get_localization_operator_tight_binding_projections[
    MBloch: BlochListMetadata,
    MState: BlochStateMetadata,
](
    states: BlochStateListWithMetadata[MBloch, MState],
) -> LocalizationOperatorWithMetadata[WannierListMetadata, MBloch]:
    projections = get_tight_binding_states(states)
    # Better performace if we provide the projection in transformed basis
    # TODO: we still want it to be sparse I think??
    converted = projections.with_state_basis(basis_k)

    return get_localization_operator_for_projections(states, converted)


def localize_tight_binding_projection[
    MBloch: BlochListMetadata,
    MState: BlochStateMetadata,
](
    states: BlochStateListWithMetadata[MBloch, MState],
) -> WannierStateListWithMetadata[WannierListMetadata, MState]:
    """
    Given a wavepacket, localize using a tight binding projection.

    Parameters
    ----------
    wavepacket : Wavepacket[_B0Inv, _B0Inv]

    Returns
    -------
    Wavepacket[_B0Inv, _B0Inv]
    """
    operator = get_localization_operator_tight_binding_projections(states)
    return localize_states(states, operator)


from slate_core import basis


def get_single_point_state_for_wavepacket[
    MBloch: BlochListMetadata,
    MState: BlochStateMetadata,
](
    state: StateWithMetadata[MState],
    idx: int | tuple[int, ...] = 0,
    origin: tuple[int, ...] | None = None,
) -> StateWithMetadata[MState]:
    state_0 = state.with_basis(basis.as_fundamental(state.basis)).assert_ok()

    if origin is None:
        idx_0 = array.max_arg(state_0)

        origin = wrap_index_around_origin(state_0["basis"], idx_0)
    return get_single_point_state_vector_excact(
        state_0["basis"], util.get_flat_index(origin, mode="wrap")
    )


def localize_single_point_projection[
    MBloch: BlochListMetadata,
    MState: BlochStateMetadata,
](
    state: StateWithMetadata[MState],
    idx: int | tuple[int, ...] = 0,
) -> WannierStateListWithMetadata[WannierListMetadata, MState]:
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
    projection = get_single_point_state_for_wavepacket(state, idx)
    # Will have better performace if we provide it in a truncated position basis
    return localize_state_projection(state, projection)


def get_exponential_state[
    MBloch: BlochListMetadata,
    MState: BlochStateMetadata,
](
    # TODO: single band version?
    states: BlochStateListWithMetadata[MBloch, MState],
    idx: int | tuple[int, ...] = 0,
    origin: int | tuple[int, ...] | None = None,
) -> StateWithMetadata[MState]:
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


def _get_exponential_decay_state[
    SB0: StackedBasisLike,
    SBV0: StackedBasisWithVolumeLike,
](
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


def localize_exponential_decay_projection[
    SB0: StackedBasisLike,
    SBV0: StackedBasisWithVolumeLike,
](
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
    return localize_state_projection(wavepacket, projection)


def get_gaussian_states[SB0: StackedBasisLike, SBV0: StackedBasisWithVolumeLike](
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


def localize_wavepacket_gaussian_projection[
    SB0: StackedBasisLike,
    SBV0: StackedBasisWithVolumeLike,
](
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
    return localize_state_projection(wavepacket, projection)


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
