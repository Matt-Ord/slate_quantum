from __future__ import annotations

import re
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    TypedDict,
    cast,
    overload,
)

import numpy as np
import scipy.ndimage  # type:ignore lib
from slate_core import BasisMetadata, TupleMetadata, basis, metadata
from slate_core.basis import (
    AsUpcast,
)

from slate_quantum.bloch._transposed_basis import BlochStateMetadata
from slate_quantum.bloch.build import (
    fraction_metadata_from_bloch_state,
    get_sample_fractions,
)
from slate_quantum.state._state import State, StateWithMetadata
from slate_quantum.wannier._localization_operator import (
    BlochListMetadata,
    LocalizationOperatorWithMetadata,
    WannierListMetadata,
    WannierStateListWithMetadata,
    localize_states,
)
from slate_quantum.wannier._projection import get_state_projections_many_band

if TYPE_CHECKING:
    from slate_core.metadata import SpacedVolumeMetadata, VolumeMetadata

    from slate_quantum.bloch.build import StackedBlockedFractionMetadata
    from slate_quantum.wannier._localization_operator import BlochStateListWithMetadata


SymmetryOp = Callable[
    [np.ndarray[Any, np.dtype[np.int_]], tuple[int, ...]],
    np.ndarray[Any, np.dtype[np.int_]],
]


class ProjectionsBasis[B0: BasisLike](TypedDict):
    basis: TupleBasisLike[_B0]


@dataclass
class Wannier90Options[B0: BasisLike]:
    projection: ProjectionsBasis[_B0] | LegacyStateVectorList[_B0, Any]
    num_iter: int = 10000
    convergence_window: int = 3
    convergence_tolerance: float = 1e-10
    symmetry_operations: Sequence[SymmetryOp] | None = None
    ignore_axes: tuple[int, ...] = ()
    """Axes which one should assume are tightly bound"""


def _build_real_lattice_block(
    meta: VolumeMetadata,
) -> str:
    # Wannier90 expects the wavefunction to be 3D, so we add the extra fake axis here
    delta_x = metadata.volume.fundamental_stacked_delta_x(meta)
    delta_x_padded = np.eye(3)
    n_dim = len(delta_x)
    delta_x_padded[:n_dim, :n_dim] = delta_x * 10**10

    return f"""begin unit_cell_cart
{"\n".join(" ".join(str(x) for x in o) for o in delta_x_padded)}
end unit_cell_cart"""


# ! cSpell:disable
def _build_k_points_block(
    # TODO: what should the bloch metadata be here?
    # Do we want to infer from the state basis instead?
    metadata: StackedBlockedFractionMetadata,
) -> str:
    fractions = get_sample_fractions(metadata)
    fractions_padded = np.zeros((3, fractions[0].shape[0]))
    fractions_padded[: metadata.n_dim] = fractions

    # mp_grid is 1 in the directions not included in the wavefunction
    mp_grid = np.ones(3, dtype=np.int_)
    mp_grid[: metadata.n_dim] = np.array(metadata.shape)

    return f"""mp_grid : {" ".join(str(x) for x in mp_grid)}
begin kpoints
{"\n".join([f"{f0!r} {f1!r} {f2!r}" for (f0, f1, f2) in fractions_padded.T])}
end kpoints"""


def _build_win_file[MBloch: BlochListMetadata, MState: BlochStateMetadata](
    states: BlochStateListWithMetadata[MBloch, MState],
    *,
    postproc_setup: bool = False,
    options: Wannier90Options[Any],
) -> str:
    """
    Build a postproc setup file.

    Parameters
    ----------
    wavepacket : Wavepacket[_B0Inv, _B0Inv]

    Returns
    -------
    str
    """
    has_projections = options.projection.get("data", None) is not None
    list_metadata = states.basis.metadata()
    sample_metadata = fraction_metadata_from_bloch_state(list_metadata.children[1])
    n_wannier = metadata.size_from_nested_shape(
        list_metadata.children[0].children[0].fundamental_shape
    )
    return f"""
{"postproc_setup = .true." if postproc_setup else ""}
num_iter = {options.num_iter}
conv_tol = {options.convergence_tolerance}
conv_window = {options.convergence_window}
write_u_matrices = .true.
{"auto_projections = .true." if has_projections else "use_bloch_phases = .true."}
num_wann = {n_wannier}
{_build_real_lattice_block(list_metadata.children[1])}
{_build_k_points_block(sample_metadata)}
search_shells = 500
"""


# ! cSpell:enable


def _get_offset_bloch_state[SB0: SpacedVolumeMetadata](
    state: StateWithMetadata[SB0], offset: tuple[int, ...]
) -> StateWithMetadata[SB0]:
    """
    Get the bloch state corresponding to the bloch k offset by 'offset'.

    Since the wavefunction is identical, this is just the rolled state vector

    Parameters
    ----------
    state : StateVector[_B0Inv]
        _description_
    offset : tuple[int, ...]
        _description_

    Returns
    -------
    StateVector[_B0Inv]
        _description_
    """
    # Note: requires state in k basis
    state_in_k = state.with_basis(basis.as_transformed(state.basis)).assert_ok()
    padded_shape = np.ones(3, dtype=np.int_)
    padded_shape[: len(state_in_k.basis.shape)] = state_in_k.basis.shape
    vector = np.roll(
        (state_in_k.raw_data).reshape(padded_shape),
        tuple(-o for o in offset),
        (0, 1, 2),
    )

    # !f_0 = (vector.shape[0] + 1) // 2
    # !f_1 = f_0 + offset[0]
    # !vector[min(f_0, f_1) : max(f_0, f_1) :, :, :] = 0

    # !f_0 = (vector.shape[1] + 1) // 2
    # !f_1 = f_0 + offset[1]
    # !vector[:, min(f_0, f_1) : max(f_0, f_1) :, :] = 0

    # !f_0 = (vector.shape[2] + 1) // 2
    # !f_1 = f_0 + offset[2]
    # !vector[:, :, min(f_0, f_1) : max(f_0, f_1) :] = 0
    upcast = AsUpcast(state_in_k.basis, state.basis.metadata())
    return State.build(upcast, vector).assert_ok()
    # Should be -offset if psi(k+b) = psi(k)
    vector = scipy.ndimage.shift(
        (state["data"]).reshape(padded_shape),
        tuple(-o for o in offset),
        mode="constant",
        cval=0,  # cSpell: disable-line
    ).reshape(-1)
    return {"basis": state["basis"], "data": vector}


def _build_mmn_file_block[MBloch: BlochListMetadata, MState: BlochStateMetadata](
    states: BlochStateListWithMetadata[MBloch, MState],
    k: tuple[int, int, int, int, int],
    *,
    options: Wannier90Options[Any],
) -> str:
    k_0, k_1, *offset = k
    block = f"{k_0} {k_1} {offset[0]} {offset[1]} {offset[2]}"
    for i in options.ignore_axes:
        offset[i] = 0

    for wavepacket_n in states:
        for wavepacket_m in states:
            mat = legacy_calculate_inner_product(
                _get_offset_bloch_state(
                    get_bloch_state_vector(wavepacket_n, k_1 - 1),
                    tuple(offset),
                ),
                as_legacy_dual_vector(get_bloch_state_vector(wavepacket_m, k_0 - 1)),
            )
            block += f"\n{np.real(mat)!r} {np.imag(mat)!r}"
    return block


def _parse_nnk_points_file(
    nnk_points_file: str,
) -> tuple[int, list[tuple[int, int, int, int, int]]]:
    block = re.search(
        r"begin nnkpts((.|\n)+?)end nnkpts",
        nnk_points_file,  # cSpell:disable-line
    )
    if block is None:
        msg = "Unable to find nnk_points block"
        raise Exception(msg)  # noqa: TRY002
    lines = block.group(1).strip().split("\n")
    first_element = int(lines[0])
    data: list[tuple[int, int, int, int, int]] = []

    for line in lines[1:]:
        values = line.split()
        data_entry = tuple(map(int, values[:5]))
        data.append(data_entry)  # type: ignore[arg-type]

    return first_element, data


def _build_mmn_file[MBloch: BlochListMetadata, MState: BlochStateMetadata](
    states: BlochStateListWithMetadata[MBloch, MState],
    nnk_points_file: str,
    *,
    options: Wannier90Options[Any],
) -> str:
    """
    Given a .nnkp file, generate the mmn file.  # cSpell:disable-line.

    Parameters
    ----------
    wavepacket : Wavepacket[_B0Inv, _B0Inv]
    file : str

    Returns
    -------
    Wavepacket[_B0Inv, _B0Inv]
    """
    meta = states.basis.metadata()
    n_wavefunctions = meta.children[0].children[0].fundamental_size
    n_k_points = meta.children[0].children[1].fundamental_size
    (n_n_tot, nnk_points) = _parse_nnk_points_file(nnk_points_file)

    newline = "\n"
    return f"""
{n_wavefunctions} {n_k_points} {n_n_tot}
{newline.join(_build_mmn_file_block(states, k, options=options) for k in nnk_points)}"""


def _build_amn_file[MBloch: BlochListMetadata, MState: BlochStateMetadata](
    states: BlochStateListWithMetadata[MBloch, MState],
    projections: LegacyStateVectorList[_B1, _B2],
) -> str:
    n_projections = projections["basis"][0].size
    n_wavefunctions = wavepackets["basis"][0][0].n
    n_k_points = wavepackets["basis"][0][1].n
    coefficients = np.array(
        [
            get_state_projections_many_band(
                get_states_at_bloch_idx(wavepackets, idx), projections
            )["data"]
            for idx in range(n_k_points)
        ]
    )
    stacked = coefficients.reshape(n_k_points, n_wavefunctions, n_projections)
    newline = "\n"
    newline.join(
        f"{m + 1} {n + 1} {k + 1} {np.real(s)} {np.imag(s)}"
        for ((k, m, n), s) in np.ndenumerate(stacked)
    )
    return f"""
{n_wavefunctions} {n_k_points} {n_projections}
{
        newline.join(
            f"{m + 1} {n + 1} {k + 1} {np.real(s)!r} {np.imag(s)!r}"
            for ((k, m, n), s) in np.ndenumerate(stacked)
        )
    }
"""


def x0_symmetry_op(idx: int, shape: tuple[int, ...], axis: int) -> int:
    idx_stacked = list(np.unravel_index(idx, shape))
    idx_stacked[axis] = -idx_stacked[axis]
    return np.ravel_multi_index(tuple(idx_stacked), shape, mode="wrap")


def x0x1_symmetry_op(idx: int, shape: tuple[int, ...], axes: tuple[int, int]) -> int:
    idx_stacked = list(np.unravel_index(idx, shape))
    idx_0 = idx_stacked[axes[0]]
    idx_stacked[axes[0]] = idx_stacked[axes[1]]
    idx_stacked[axes[1]] = idx_0
    return np.ravel_multi_index(tuple(idx_stacked), shape, mode="wrap")


def _get_fundamental_k_points(
    metadata: TupleMetadata[tuple[BasisMetadata, ...], None],
    symmetry: Sequence[SymmetryOp],
) -> np.ndarray[tuple[int], np.dtype[np.int_]]:
    fundamental_idx = np.arange(metadata.fundamental_size)
    for op in symmetry:
        b = op(fundamental_idx, metadata.shape)
        fundamental_idx = np.minimum(b, fundamental_idx)

    return fundamental_idx


def _build_dmn_file[MBloch: BlochListMetadata, MState: BlochStateMetadata](
    states: BlochStateListWithMetadata[MBloch, MState],
    symmetry: Sequence[SymmetryOp],
) -> str:
    n_wavefunctions = states.basis.metadata().children[0].children[0].fundamental_size
    n_k_points = states.basis.metadata().children[0].children[1].fundamental_size
    n_symmetry = len(symmetry)

    fundamental = _get_fundamental_k_points(
        states.basis.metadata().children[0].children[1], symmetry
    )
    u, idx = np.unique(fundamental, return_inverse=True)
    n_k_points_irr = u.size

    idx_padded = np.pad(idx, (0, idx.size % 10), "constant", constant_values=-1)
    u_padded = np.pad(u, (0, idx.size % 10), "constant", constant_values=-1)
    raise NotImplementedError
    newline = "\n"
    return f"""
{n_wavefunctions} {n_symmetry} {n_k_points_irr} {n_k_points}

{
        newline.join(
            " ".join((j + 1) for j in i if j != -1) for i in idx_padded.reshape(10, -1)
        )
    }
{
        newline.join(
            " ".join((j + 1) for j in i if j != -1) for i in u_padded.reshape(10, -1)
        )
    }
"""


def _parse_u_mat_file_block(
    block: list[str],
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
    variables = [line.strip().split() for line in block]
    return np.array([float(s[0]) + 1j * float(s[1]) for s in variables])  # type: ignore[no-any-return]


def _get_localization_operator_from_u_mat_file[
    MBloch: BlochListMetadata,
](
    wavepackets_basis: MBloch,
    u_matrix_file: str,
) -> LocalizationOperatorWithMetadata[WannierListMetadata, MBloch]:
    """
    Get the u_mat coefficients, indexed such that U_{m,n}^k === out[n, m, k].

    Parameters
    ----------
    u_matrix_file : str

    Returns
    -------
    np.ndarray[tuple[int, int, int], np.dtype[np.complex_]]
    """
    lines = u_matrix_file.splitlines()
    n_wavefunctions = int(lines[1].strip().split()[1])
    n_bands = int(lines[1].strip().split()[2])
    a = np.array(
        [
            [
                _parse_u_mat_file_block(
                    lines[
                        4 + n + n_wavefunctions * m :: 2 + (n_wavefunctions * n_bands)
                    ]
                )
                for n in range(n_wavefunctions)
            ]
            for m in range(n_bands)
        ]
    )
    return {
        "basis": TupleBasis(
            wavepackets_basis[1],
            VariadicTupleBasis((projection_basis, wavepackets_basis[0]), None),
        ),
        "data": np.moveaxis(a, -1, 0).reshape(-1),
    }


def _write_setup_files_wannier90[
    B0: BasisLike,
    SB0: StackedBasisLike,
    SBL0: StackedBasisWithVolumeLike,
    B1: BasisLike,
](
    wavepackets: BlochWavefunctionListList[_B0, _SB0, _SBL0],
    tmp_dir_path: Path,
    *,
    options: Wannier90Options[_B1],
) -> None:
    win_filename = tmp_dir_path / "spa.win"
    with win_filename.open("w") as f:
        f.write(_build_win_file(wavepackets, postproc_setup=True, options=options))


def _write_localization_files_wannier90[
    MBloch: BlochListMetadata,
    MState: BlochStateMetadata,
](
    states: BlochStateListWithMetadata[MBloch, MState],
    tmp_dir_path: Path,
    n_nkp_file: str,
    *,
    options: Wannier90Options[_B1],
) -> None:
    win_filename = tmp_dir_path / "spa.win"
    with win_filename.open("w") as f:
        f.write(_build_win_file(wavepackets, options=options))
    converted = convert_state_vector_list_to_basis(
        wavepackets,
        TupleBasis(
            *tuple(
                basis_as_transformed_basis(axis)
                if idx not in options.ignore_axes
                else axis
                for (idx, axis) in enumerate(wavepackets["basis"][1])
            )
        ),
    )

    mmn_filename = tmp_dir_path / "spa.mmn"
    with mmn_filename.open("w") as f:
        f.write(_build_mmn_file(converted, n_nkp_file, options=options))
    if options.projection.get("data", None) is not None:
        projection = cast("LegacyStateVectorList[_B1, Any]", options.projection)
        amn_filename = tmp_dir_path / "spa.amn"
        with amn_filename.open("w") as f:
            f.write(_build_amn_file(converted, projection))

    if options.symmetry_operations is not None:
        dmn_filename = tmp_dir_path / "spa.dmn"
        with dmn_filename.open("w") as f:
            f.write(_build_dmn_file(converted, options.symmetry_operations))


# ruff: noqa: S603, S607, S108
def _run_wannier90(directory: Path) -> None:
    subprocess.run(
        [
            "wannier90.x",
            "spa.win",
        ],
        check=True,
        cwd=directory,
    )


# ruff: noqa: S603, S607, S108
def _run_wannier90_in_container(directory: Path) -> None:
    container = "f776014440144052e94a86044eaf6ee4ce131f80ae12df0bd0d2bdcf24206bfa"
    subprocess.run(
        ["docker", "exec", container, "rm", "-r", "/tmp/scratch_w90"],
        check=False,
    )
    subprocess.run(
        ["docker", "exec", container, "mkdir", "/tmp/scratch_w90"],
        check=True,
    )
    subprocess.run(
        ["docker", "cp", ".", f"{container}:/tmp/scratch_w90"],
        check=True,
        cwd=directory,
    )
    subprocess.run(
        [
            "docker",
            "exec",
            "-w",
            "/tmp/scratch_w90",
            container,
            "wannier90.x",
            "spa.win",
        ],
        check=True,
    )
    subprocess.run(
        ["docker", "cp", f"{container}:/tmp/scratch_w90/.", "."],
        check=True,
        cwd=directory,
    )
    subprocess.run(
        ["docker", "exec", container, "rm", "-r", "/tmp/scratch_w90"],
        check=True,
    )


@overload
def get_localization_operator_wannier90[
    B0: BasisLike,
    SB0: StackedBasisLike,
    SBL0: StackedBasisWithVolumeLike,
    B1: BasisLike,
](
    wavefunctions: BlochWavefunctionListList[_B0, _SB0, _SBL0],
    *,
    options: Wannier90Options[_B1],
) -> LocalizationOperator[_SB0, _B1, _B0]: ...


@overload
def get_localization_operator_wannier90[
    B0: BasisLike,
    SB0: StackedBasisLike,
    SBL0: StackedBasisWithVolumeLike,
](
    wavefunctions: BlochWavefunctionListList[_B0, _SB0, _SBL0],
    *,
    options: None = None,
) -> LocalizationOperator[_SB0, FundamentalBasis[BasisMetadata], _B0]: ...


def get_localization_operator_wannier90[
    MBloch: BlochListMetadata,
    MState: BlochStateMetadata,
](
    states: BlochStateListWithMetadata[MBloch, MState],
    *,
    options: Wannier90Options[Any] | None = None,
    # TODO: what metadata should we use here?
) -> LocalizationOperatorWithMetadata[WannierListMetadata, MBloch]:
    """
    Localizes a set of wavepackets using wannier 90.

    Note this requires a user to manually run wannier90 and input the resulting wannier90 nnk_pts and _u.mat file
    into the temporary directory created by the function.

    Also requires the wavepackets to have their samples in the fundamental basis

    Parameters
    ----------
    wavepackets : WavepacketList[_B0, _SB0, _SBL0]
        Wavepackets to localize
    projections : StateVectorList[_B1, _B2]
        Projections used in the localization procedure

    Returns
    -------
    WavepacketList[_B1, _SB0, _SBL0]
        Localized wavepackets, with each wavepacket corresponding to a different projection
    """
    options = (
        Wannier90Options[FundamentalBasis[BasisMetadata]](
            projection={
                "basis": TupleBasis(
                    FundamentalBasis(wavefunctions["basis"][0][0].n),
                ),
            },
        )
        if options is None
        else options
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        # Build Files for initial Setup
        _write_setup_files_wannier90(states, tmp_dir_path, options=options)
        try:
            _run_wannier90(tmp_dir_path)
        except Exception:  # noqa: BLE001
            input(f"Run Wannier 90 in {tmp_dir_path}")

        # Load Setup Files
        n_nkp_filename = tmp_dir_path / "spa.nnkp"  # cSpell:disable-line
        with n_nkp_filename.open("r") as f:
            n_nkp_file = f.read()

        # Build Files for localisation
        _write_localization_files_wannier90(
            states,
            tmp_dir_path,
            n_nkp_file,
            options=options,  # type: ignore should be fundamental basis, but we have no way of ensuring this in the type system
        )
        try:
            _run_wannier90(tmp_dir_path)
        except Exception:  # noqa: BLE001
            input(f"Run Wannier 90 in {tmp_dir_path}")

        # Load Result files, and localize wavepacket
        u_mat_filename = tmp_dir_path / "spa_u.mat"
        with u_mat_filename.open("r") as f:
            u_mat_file = f.read()

    return _get_localization_operator_from_u_mat_file(
        states.basis.metadata().children[0], u_mat_file
    )


def localize_wavepacket_wannier90[
    MBloch: BlochListMetadata,
    MState: BlochStateMetadata,
](
    states: BlochStateListWithMetadata[MBloch, MState],
    *,
    options: Wannier90Options[Any],
) -> WannierStateListWithMetadata[WannierListMetadata, MState]:
    """
    Localizes a set of wavepackets using wannier 90, with a single point projection as an initial guess.

    Note this requires a user to manually run wannier90 and input the resulting wannier90 nnk_pts and _u.mat file
    into the temporary directory created by the function.

    Also requires the wavepackets to have their samples in the fundamental basis

    Parameters
    ----------
    wavepackets : WavepacketList[_B0, _SB0, _SBL0]
        Wavepackets to localize
    projections : StateVectorList[_B1, _B2]
        Projections used in the localization procedure

    Returns
    -------
    WavepacketList[_B1, _SB0, _SBL0]
        Localized wavepackets, with each wavepacket correspondign to a different projection
    """
    operator = get_localization_operator_wannier90(states, options=options)
    return localize_states(states, operator)


def get_localization_operator_wannier90_individual_bands[
    MBloch: BlochListMetadata,
    MState: BlochStateMetadata,
](
    states: BlochStateListWithMetadata[MBloch, MState],
    # Specifically, it's diagonal
) -> LocalizationOperatorWithMetadata[WannierListMetadata, MBloch]:
    options = Wannier90Options[FundamentalBasis[BasisMetadata]](
        projection={"basis": VariadicTupleBasis((FundamentalBasis(1), None))},
        convergence_tolerance=1e-20,
        ignore_axes=(2,),
    )
    operator_data = [
        get_localization_operator_wannier90(
            as_wavepacket_list([wavepacket]), options=options
        )["data"]
        for wavepacket in wavepacket_list_into_iter(wavepackets)
    ]
    n_bands = wavepackets["basis"][0][0].n
    out = np.zeros(
        (wavepackets["basis"][0][1].n, n_bands, n_bands), dtype=np.complex128
    )
    out[:, np.arange(n_bands), np.arange(n_bands)] = (
        np.array(operator_data)
        .reshape(n_bands, wavepackets["basis"][0][1].n)
        .swapaxes(0, 1)
    )
    return {
        "basis": TupleBasis(
            wavepackets["basis"][0][1],
            VariadicTupleBasis(
                (wavepackets["basis"][0][0], wavepackets["basis"][0][0]), None
            ),
        ),
        "data": out.reshape(-1),
    }
