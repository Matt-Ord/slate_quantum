from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.constants import hbar  # type: ignore lib
from slate.basis.stacked import DiagonalBasis, as_tuple_basis, diagonal_basis
from slate.basis.transformed import fundamental_transformed_tuple_basis_from_metadata
from slate.metadata.stacked.volume import (
    fundamental_stacked_dk,
    fundamental_stacked_k_points,
)

from slate_quantum.model.operator._operator import Operator, SplitOperator

if TYPE_CHECKING:
    from slate.metadata import Metadata2D, VolumeMetadata
    from slate.metadata.stacked import StackedMetadata

    from slate_quantum.model.operator._potential import Potential


def build_kinetic_energy_operator(
    metadata: StackedMetadata[Any, Any],
    mass: float,
    bloch_fraction: np.ndarray[Any, np.dtype[np.float64]] | None = None,
) -> Operator[Any, Any, DiagonalBasis[np.complex128, Any, Any, None]]:
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
        diagonal_basis((momentum_basis, momentum_basis.conjugate_basis()), None), energy
    )


def build_kinetic_hamiltonian(
    potential: Potential[VolumeMetadata, np.complex128],
    mass: float,
    bloch_fraction: np.ndarray[Any, np.dtype[np.float64]] | None = None,
) -> Operator[
    Metadata2D[VolumeMetadata, VolumeMetadata, None],
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
    potential_hamiltonian = potential.with_basis(basis)
    kinetic_hamiltonian = build_kinetic_energy_operator(
        basis.metadata(), mass, bloch_fraction
    )
    n = basis.size
    return SplitOperator(
        basis,
        potential_hamiltonian.raw_data,
        np.ones((n,), dtype=np.complex128),
        kinetic_hamiltonian.raw_data,
        np.ones((n,), dtype=np.complex128),
    )
