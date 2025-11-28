from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.constants import hbar  # type: ignore lib
from slate_core import Basis, TupleBasis, TupleMetadata
from slate_core.basis import (
    AsUpcast,
    SplitBasis,
    transformed_from_metadata,
    trigonometric_transformed_from_metadata,
)
from slate_core.metadata import (
    fundamental_nk_points,
    fundamental_nx_points,
)
from slate_core.metadata.volume import (
    fundamental_stacked_delta_k,
    fundamental_stacked_dk,
)

from slate_quantum.operator._diagonal import momentum_operator_basis
from slate_quantum.operator._operator import Operator, OperatorBasis, OperatorMetadata

if TYPE_CHECKING:
    from slate_core import Ctype
    from slate_core.metadata import (
        EvenlySpacedLengthMetadata,
    )
    from slate_core.metadata.volume import AxisDirections, EvenlySpacedVolumeMetadata


def _wrap_k_points(
    k_points: np.ndarray[Any, np.dtype[np.floating]], delta_k: tuple[float, ...]
) -> np.ndarray[Any, np.dtype[np.floating]]:
    """Wrap values into the range [-delta_k/2, delta_k/2]."""
    shift = np.array(delta_k).reshape(-1, 1) / 2
    return np.mod(k_points + shift, 2 * shift) - shift


def _get_bloch_phase(
    metadata: TupleMetadata[tuple[EvenlySpacedLengthMetadata, ...], AxisDirections],
    bloch_fraction: np.ndarray[Any, np.dtype[np.floating]] | None = None,
) -> np.ndarray[Any, np.dtype[np.floating]]:
    """Get the default Bloch fraction for the given metadata."""
    bloch_fraction = (
        np.zeros(len(metadata.children)) if bloch_fraction is None else bloch_fraction
    )
    return np.tensordot(fundamental_stacked_dk(metadata), bloch_fraction, axes=(0, 0))


# TODO: we might want to split this into a  # noqa: FIX002
# Get derivative matrix and get derivative squared matrix.
def _fundamental_n_kinetic(
    metadata: EvenlySpacedLengthMetadata,
) -> np.ndarray[Any, np.dtype[np.int_]]:
    """Get the k^2 eigenvalues."""
    if metadata.interpolation == "Fourier":
        return fundamental_nk_points(metadata)
    return fundamental_nx_points(metadata)


def _fundamental_stacked_n_kinetic(
    metadata: TupleMetadata[tuple[EvenlySpacedLengthMetadata, ...], AxisDirections],
) -> tuple[np.ndarray[Any, np.dtype[np.int_]], ...]:
    mesh = np.meshgrid(
        *(_fundamental_n_kinetic(m) for m in metadata.children),
        indexing="ij",
    )
    return tuple(nki.ravel() for nki in mesh)


def _fundamental_stacked_kinetic_points(
    metadata: TupleMetadata[tuple[EvenlySpacedLengthMetadata, ...], AxisDirections],
    bloch_phase: np.ndarray[Any, np.dtype[np.floating]],
) -> tuple[np.ndarray[Any, np.dtype[np.floating]], ...]:
    """Get the stacked k^2 eigenvalues."""
    points = cast(
        "np.ndarray[tuple[int, int], np.dtype[np.floating]]",
        np.einsum(  # type: ignore unknown
            "ij,ik->jk",
            fundamental_stacked_dk(metadata),
            _fundamental_stacked_n_kinetic(metadata),
        ),
    )
    points += bloch_phase[:, np.newaxis]
    # In a periodic basis we need to wrap the k points to the first Brillouin zone.
    are_periodic = [c.interpolation == "Fourier" for c in metadata.children]
    points[are_periodic] = _wrap_k_points(
        points[are_periodic],
        tuple(
            np.linalg.norm(fundamental_stacked_delta_k(metadata), axis=1)[are_periodic]
        ),
    )
    return tuple(points)


type NestedBool = bool | tuple[NestedBool, ...]


def kinetic_basis[M: EvenlySpacedVolumeMetadata](
    metadata: M, *, is_dual: NestedBool | None = None
) -> Basis[M, Ctype[np.complexfloating]]:
    """Get a transformed fundamental basis with the given metadata."""
    is_dual = (
        is_dual
        if isinstance(is_dual, tuple)
        else tuple(False for _ in metadata.children)
    )

    children = tuple(
        transformed_from_metadata(c, is_dual=dual)
        if c.interpolation == "Fourier"
        else trigonometric_transformed_from_metadata(c, is_dual=dual)
        for (c, dual) in zip(metadata.children, is_dual, strict=False)
    )
    return AsUpcast(
        TupleBasis(children, metadata.extra).resolve_ctype(), metadata
    ).resolve_ctype()


def _kinetic_energy_with_phase_offset[M: EvenlySpacedVolumeMetadata](
    metadata: M,
    mass: float,
    bloch_phase: np.ndarray[Any, np.dtype[np.floating]],
) -> Operator[OperatorBasis[M], np.dtype[np.complexfloating]]:
    """Given a mass and a basis calculate the kinetic part of the Hamiltonian."""
    k_points = _fundamental_stacked_kinetic_points(metadata, bloch_phase=bloch_phase)
    energy = cast(
        "np.ndarray[Any, np.dtype[np.complexfloating]]",
        np.sum(
            np.square(hbar * np.asarray(k_points)) / (2 * mass),
            axis=0,
            dtype=np.complex128,
        ),
    )
    momentum_basis = kinetic_basis(metadata)

    return Operator(
        AsUpcast(
            momentum_operator_basis(momentum_basis),
            TupleMetadata((metadata, metadata)),
        ),
        energy,
    )


def kinetic_energy[M: EvenlySpacedVolumeMetadata](
    metadata: M,
    mass: float,
    bloch_fraction: np.ndarray[Any, np.dtype[np.floating]] | None = None,
) -> Operator[OperatorBasis[M], np.dtype[np.complexfloating]]:
    """Given a mass and a basis calculate the kinetic part of the Hamiltonian."""
    bloch_phase = _get_bloch_phase(metadata, bloch_fraction)
    return _kinetic_energy_with_phase_offset(metadata, mass, bloch_phase)


def kinetic_hamiltonian[M: EvenlySpacedVolumeMetadata](
    potential: Operator[OperatorBasis[M], np.dtype[np.complexfloating]],
    mass: float,
    bloch_fraction: np.ndarray[Any, np.dtype[np.floating]] | None = None,
) -> Operator[
    AsUpcast[
        SplitBasis[OperatorBasis[M], OperatorBasis[M]],
        OperatorMetadata[M],
    ],
    np.dtype[np.complexfloating],
]:
    """Calculate the total hamiltonian in momentum basis for a given potential and mass."""
    metadata = potential.basis.metadata().children[0]
    kinetic_hamiltonian = kinetic_energy(metadata, mass, bloch_fraction)
    basis = SplitBasis(potential.basis, kinetic_hamiltonian.basis)

    return Operator(
        AsUpcast(basis, TupleMetadata((metadata, metadata))),
        np.concatenate([potential.raw_data, kinetic_hamiltonian.raw_data], axis=None),
    )
