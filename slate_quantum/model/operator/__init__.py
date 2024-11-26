"""Representation of Quantum Operators."""

from __future__ import annotations

from ._operator import Operator, OperatorList
from ._super_operator import SuperOperator, SuperOperatorMetadata
from .build import build_kinetic_energy_operator, build_kinetic_hamiltonian
from .linalg import into_diagonal, into_diagonal_hermitian
from .potential import Potential, build_cos_potential, repeat_potential

__all__ = [
    "Operator",
    "OperatorList",
    "Potential",
    "SuperOperator",
    "SuperOperatorMetadata",
    "build_cos_potential",
    "build_kinetic_energy_operator",
    "build_kinetic_hamiltonian",
    "into_diagonal",
    "into_diagonal_hermitian",
    "repeat_potential",
]
