from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.constants import hbar  # type: ignore lib
from slate.basis import (
    DiagonalBasis,
    SplitBasis,
    TupleBasis,
    as_tuple_basis,
    diagonal_basis,
    fundamental_transformed_tuple_basis_from_metadata,
)
from slate.metadata.volume import (
    fundamental_stacked_dk,
    fundamental_stacked_k_points,
)

from slate_quantum.model.operator._operator import Operator

if TYPE_CHECKING:
    from slate.metadata import (
        Metadata2D,
        SpacedLengthMetadata,
        SpacedVolumeMetadata,
        StackedMetadata,
    )
    from slate.metadata.volume import AxisDirections

    from slate_quantum.model.operator.potential._potential import Potential


def build_kinetic_energy_operator[M: SpacedLengthMetadata, E: AxisDirections](
    metadata: StackedMetadata[M, E],
    mass: float,
    bloch_fraction: np.ndarray[Any, np.dtype[np.float64]] | None = None,
) -> Operator[
    Metadata2D[StackedMetadata[M, E], StackedMetadata[M, E], Any],
    Any,
    DiagonalBasis[
        np.complex128,
        TupleBasis[M, E, np.complex128],
        TupleBasis[M, E, np.complex128],
        None,
    ],
]:
    """
    Given a mass and a basis calculate the kinetic part of the Hamiltonian.

    Parameters
    ----------
    basis : _B0Inv
    mass : float
    bloch_fraction : np.ndarray[tuple[int], np.dtype[np.float_]] | None, optional
        bloch phase, by default None

    Returns
    -------
    Hamiltonian[_B0Inv]
    """
    bloch_fraction = (
        np.zeros(len(metadata.children)) if bloch_fraction is None else bloch_fraction
    )

    bloch_phase = np.tensordot(
        fundamental_stacked_dk(metadata), bloch_fraction, axes=(0, 0)
    )
    k_points = fundamental_stacked_k_points(metadata) + bloch_phase[:, np.newaxis]
    energy = np.sum(
        np.square(hbar * k_points) / (2 * mass), axis=0, dtype=np.complex128
    )
    momentum_basis = fundamental_transformed_tuple_basis_from_metadata(metadata)

    return Operator(
        diagonal_basis((momentum_basis, momentum_basis.dual_basis()), None), energy
    )


def build_kinetic_hamiltonian[M: SpacedVolumeMetadata](
    potential: Potential[M, np.complex128],
    mass: float,
    bloch_fraction: np.ndarray[Any, np.dtype[np.float64]] | None = None,
) -> Operator[
    Metadata2D[M, M, None],
    np.complex128,
]:
    """
    Calculate the total hamiltonian in momentum basis for a given potential and mass.

    Parameters
    ----------
    potential : Potential[_L0, _L1, _L2]
    mass : float
    bloch_fraction : np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]

    Returns
    -------
    MomentumBasisHamiltonian[_L0, _L1, _L2]
    """
    basis = as_tuple_basis(potential.basis.inner)
    potential_hamiltonian = potential.with_basis(DiagonalBasis(basis))
    kinetic_hamiltonian = build_kinetic_energy_operator(
        basis.metadata().children[0], mass, bloch_fraction
    )

    return Operator(
        SplitBasis(potential_hamiltonian.basis, kinetic_hamiltonian.basis),
        np.concatenate(
            [potential_hamiltonian.raw_data, kinetic_hamiltonian.raw_data], axis=None
        ),
    )
