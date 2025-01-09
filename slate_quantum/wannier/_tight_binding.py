from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar, TypeVarTuple, cast, overload

import numpy as np
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.stacked_basis.conversion import (
    tuple_basis_as_fundamental,
)
from surface_potential_analysis.stacked_basis.util import (
    get_x01_mirrored_index,
    wrap_index_around_origin,
)
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_to_position_basis,
)
from surface_potential_analysis.util.decorators import timed
from surface_potential_analysis.wavepacket.conversion import (
    convert_wavepacket_to_position_basis,
)
from surface_potential_analysis.wavepacket.get_eigenstate import (
    get_wavepacket_state_vector,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    BlochWavefunctionList,
    get_wavepacket_sample_fractions,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.legacy import (
        BasisWithLengthLike,
        FundamentalBasis,
        TupleBasisLike,
        TupleBasisWithLengthLike,
    )
    from surface_potential_analysis.types import (
        ArrayIndexLike,
        SingleFlatIndexLike,
        SingleIndexLike,
        SingleStackedIndexLike,
    )

    _FB0 = TypeVar("_FB0", bound=FundamentalBasis[Any])
    _FB1 = TypeVar("_FB1", bound=FundamentalBasis[Any])

    _SB0 = TypeVar("_SB0", bound=TupleBasisLike[*tuple[Any, ...]])
    _SB1 = TypeVar("_SB1", bound=TupleBasisWithLengthLike[*tuple[Any, ...]])

    _BL0 = TypeVar("_BL0", bound=BasisWithLengthLike[Any, Any, Literal[3]])
    _BL1 = TypeVar("_BL1", bound=BasisWithLengthLike[Any, Any, Literal[3]])
    _BL2 = TypeVar("_BL2", bound=BasisWithLengthLike[Any, Any, Literal[3]])

    _TS = TypeVarTuple("_TS")


@overload
def _get_global_phases(
    wavepacket: BlochWavefunctionList[_SB0, _SB1],
    idx: SingleIndexLike,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]: ...


@overload
def _get_global_phases(
    wavepacket: BlochWavefunctionList[_SB0, _SB1],
    idx: ArrayIndexLike[*_TS],
) -> np.ndarray[tuple[int, *_TS], np.dtype[np.float64]]: ...


def _get_global_phases(
    wavepacket: BlochWavefunctionList[_SB0, _SB1],
    idx: SingleIndexLike | ArrayIndexLike[*_TS],
) -> (
    np.ndarray[tuple[int, *_TS], np.dtype[np.float64]]
    | np.ndarray[tuple[int], np.dtype[np.float64]]
):
    """
    Get the global bloch phase at a given index in the irreducible cell.

    returns a list of the global phase of each eigenstate in the wavepacket

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, TupleBasisLike[tuple[_A3d0Inv, _A3d1Inv, _A3d2Inv]]
        The wavepacket to get the global phase for
    idx : int | tuple[int, int, int], optional
        The index in ravelled or unravelled form, by default 0

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.float_]]
        phases for each sample in the wavepacket
    """
    basis = tuple_basis_as_fundamental(wavepacket["basis"][1])
    util = BasisUtil(basis)

    nx_points = idx if isinstance(idx, tuple) else util.get_stacked_index(idx)
    nx_fractions = tuple(a / ni for (a, ni) in zip(nx_points, basis.shape, strict=True))

    nk_fractions = get_wavepacket_sample_fractions(wavepacket["basis"][0])
    return 2 * np.pi * np.tensordot(nk_fractions, nx_fractions, axes=(0, 0))  # type: ignore[no-any-return]


def _get_bloch_wavefunction_phases(
    wavepacket: BlochWavefunctionList[_SB0, _SB1],
    idx: SingleIndexLike = 0,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """
    Get the phase of the bloch wavefunctions at the given point in position space.

    Parameters
    ----------
    wavepacket : Wavepacket[ _NS0Inv, _NS1Inv, TupleBasisLike[tuple[PositionBasis[_L0Inv], PositionBasis[_L1Inv], _A3d2Inv]]
        the wavepacket to calculate the phase of
    idx : SingleIndexLike, optional
        the index in real space, by default 0

    Returns
    -------
    np.ndarray[tuple[_NS0Inv, _NS1Inv], np.dtype[np.float_]]
        the angle for each point in the wavepacket
    """
    converted = convert_wavepacket_to_position_basis(wavepacket)
    util = BasisUtil(converted["basis"][1])
    idx = util.get_flat_index(idx, mode="wrap") if isinstance(idx, tuple) else idx

    return np.angle(converted["data"].reshape(converted["basis"].shape)[:, idx])  # type: ignore[return-value]


@timed
def localize_tightly_bound_wavepacket_idx(
    wavepacket: BlochWavefunctionList[_SB0, _SB1],
    idx: SingleIndexLike = 0,
    angle: float = 0,
) -> BlochWavefunctionList[_SB0, _SB1]:
    """
    Localize a wavepacket in momentum basis.

    Parameters
    ----------
    wavepacket : Wavepacket[ _NS0Inv, _NS1Inv, TupleBasisLike[tuple[ TruncatedBasis[_L0Inv, MomentumBasis[_LF0Inv]], TruncatedBasis[_L1Inv, MomentumBasis[_LF1Inv]], ExplicitBasis[_L2Inv, PositionBasis[_LF2Inv]], ], ]
    idx : SingleIndexLike , optional
        Index of the eigenstate to localize, by default 0
        This index is taken in the irreducible unit cell
    angle: float, optional
        Angle to localize the wavepacket to at the point idx.
        Each wavefunction will have the phase exp(i * angle) at the position
        given by idx

    Returns
    -------
    Wavepacket[ _NS0Inv, _NS1Inv, TupleBasisLike[tuple[ TruncatedBasis[_L0Inv, MomentumBasis[_LF0Inv]], TruncatedBasis[_L1Inv, MomentumBasis[_LF1Inv]], ExplicitBasis[_L2Inv, PositionBasis[_LF2Inv]], ], ]
    """
    bloch_angles = _get_bloch_wavefunction_phases(wavepacket, idx)
    global_phases = _get_global_phases(wavepacket, idx)

    phases = np.exp(-1j * (bloch_angles + global_phases - angle))
    old_data = wavepacket["data"].reshape(wavepacket["basis"].shape)
    localized_data = old_data * phases[:, np.newaxis]

    return {"basis": wavepacket["basis"], "data": localized_data.reshape(-1)}


def get_wavepacket_two_points(
    wavepacket: BlochWavefunctionList[
        TupleBasisLike[_FB0, _FB1, FundamentalBasis[Literal[1]]],
        TupleBasisWithLengthLike[_BL0, _BL1, _BL2],
    ],
    offset: tuple[int, int] = (0, 0),
) -> tuple[SingleStackedIndexLike, SingleStackedIndexLike]:
    """
    Get the index of the maximum, and the index mirrored in x01 for the wavepacket, wrapped about the given offset.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, _B3d0Inv]
    offset : tuple[int, int], optional
        offset of the point to use as an origin, by default (0, 0)

    Returns
    -------
    tuple[SingleStackedIndexLike, SingleStackedIndexLike]
    """
    util = BasisUtil(wavepacket["basis"][1])
    origin = (util.shape[0] * offset[0], util.shape[1] * offset[1], 0)

    converted = convert_state_vector_to_position_basis(
        get_wavepacket_state_vector(wavepacket, 0)
    )  # type: ignore[arg-type,var-annotated]
    converted_util = BasisUtil(converted["basis"])
    idx_0 = cast(  # can't infer np.argmax
        "SingleStackedIndexLike",
        converted_util.get_stacked_index(np.argmax(np.abs(converted["data"]), axis=-1)),
    )
    idx_0 = wrap_index_around_origin(wavepacket["basis"][1], idx_0, origin=origin)
    idx_1 = get_x01_mirrored_index(idx_0)
    idx_1 = wrap_index_around_origin(wavepacket["basis"][1], idx_1, origin=origin)
    return (idx_0, idx_1)


def _wrap_phases(
    phases: np.ndarray[tuple[int], np.dtype[np.float64]], half_width: float = np.pi
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    return (phases + half_width) % (2 * half_width) - half_width  # type: ignore[no-any-return]


@timed
def localize_tightly_bound_wavepacket_two_point_max(
    wavepacket: BlochWavefunctionList[
        TupleBasisLike[_FB0, _FB1, FundamentalBasis[Literal[1]]],
        TupleBasisWithLengthLike[_BL0, _BL1, _BL2],
    ],
    offset: tuple[int, int] = (0, 0),
    angle: float = 0,
) -> BlochWavefunctionList[
    TupleBasisLike[_FB0, _FB1, FundamentalBasis[Literal[1]]],
    TupleBasisLike[_BL0, _BL1, _BL2],
]:
    """
    Normalize a wavepacket using a 'two-point' calculation.

    This makes use of the x0=x1 symmetry in the wavepacket to calculate the angle of a wavepacket at the origin.
    We find that although the wavepacket may be zero at the origin, the angle changes sharply at the transition through
    the origin. Unfortunately this process doesn't work - to get the best localisation
    for a wavepacket which has a node in the origin we need to have one lobe with a phase of 0 and
    the other with a phase of pi (this produces +-pi/2 at both)
    For now we only consider the max location of the k=0 bloch wavefunction.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, _B3d0Inv]
        Initial wavepacket
    offset: tuple[int, int], optional
        offset of the point to localize, by default (0,0)
    angle : float, optional
        angle at the origin, by default 0

    Returns
    -------
    Wavepacket[_NS0Inv, _NS1Inv, _B3d0Inv]
        Normalized wavepacket
    """
    idx_0, idx_1 = get_wavepacket_two_points(wavepacket, offset)

    bloch_phases_0 = _get_bloch_wavefunction_phases(wavepacket, idx_0)
    global_phases_0 = _get_global_phases(wavepacket, idx_0)
    phi_0 = bloch_phases_0 + global_phases_0

    bloch_phases_1 = _get_bloch_wavefunction_phases(wavepacket, idx_1)
    global_phases_1 = _get_global_phases(wavepacket, idx_1)
    phi_1 = bloch_phases_1 + global_phases_1

    # Normalize such that the average distance phi_0, phi_1 is zero
    averaged_fix = (
        phi_0 + phi_1 - phi_1 + 0.5 * (_wrap_phases(phi_1 - phi_0, np.pi / 2))
    )
    phases = np.exp(-1j * (averaged_fix - angle))
    # Use this to test the convergence of both points
    # ! converted_wavepacket = convert_wavepacket_to_position_basis(wavepacket)
    # ! converted_normalized = converted_wavepacket["data"] * phases[:, :, np.newaxis]
    # ! util = BasisConfigUtil(converted_wavepacket["basis"])
    # ! mirrored_idx = util.get_flat_index(mirror_idx, mode="wrap")
    # ! p r int(mirror_idx, max_idx)
    # ! p r int(
    # !     np.average(
    # !         np.abs(
    # !             wrap_phases(
    # !                 np.angle(
    # !                     np.exp(1j * (global_phases_mirror))
    # !                     * converted_normalized[:, :, mirrored_idx]
    # !                 ),
    # !                 np.pi / 2,
    # !             )
    # !         )
    # !     )
    # ! )
    # ! p r int(
    # !     np.average(
    # !         np.abs(
    # !             np.angle(
    # !                 np.exp(1j * global_phases_max)
    # !                 * converted_normalized[
    # !                     :, :, util.get_flat_index(max_idx, mode="wrap")
    # !                 ]
    # !             )
    # !         )
    # !     )
    # ! )
    fixed_eigenvectors = wavepacket["data"] * phases[:, np.newaxis]

    return {"basis": wavepacket["basis"], "data": fixed_eigenvectors}


@timed
def localize_tightly_bound_wavepacket_max_point(
    wavepacket: BlochWavefunctionList[_SB0, _SB1],
    angle: float = 0,
) -> BlochWavefunctionList[_SB0, _SB1]:
    """
    Normalize a wavepacket using a 'single-point' calculation.

    This makes use of the x0=x1 symmetry in the wavepacket, and the fact that the jump in phase over the symmetry is 0 or pi.
    We therefore only need to localize one of the maximum points, and we get the other for free

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, _B3d0Inv]
        Initial wavepacket
    angle : float, optional
        angle at the origin, by default 0

    Returns
    -------
    Wavepacket[_NS0Inv, _NS1Inv, _B3d0Inv]
        Normalized wavepacket
    """
    converted = convert_state_vector_to_position_basis(
        get_wavepacket_state_vector(wavepacket, 0)
    )
    max_idx = cast("SingleFlatIndexLike", np.argmax(np.abs(converted["data"]), axis=-1))
    max_idx = BasisUtil(converted["basis"]).get_stacked_index(max_idx)
    max_idx = wrap_index_around_origin(wavepacket["basis"][1], max_idx, axes=(0, 1))

    bloch_phases = _get_bloch_wavefunction_phases(wavepacket, max_idx)
    global_phases = _get_global_phases(wavepacket, max_idx)

    phases = np.exp(-1j * (bloch_phases + global_phases - angle))
    fixed_eigenvectors = wavepacket["data"] * phases[:, :, np.newaxis]

    return {"basis": wavepacket["basis"], "data": fixed_eigenvectors}
