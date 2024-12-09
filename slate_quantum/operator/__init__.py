"""Representation of Quantum Operators."""

from __future__ import annotations

from slate_quantum.operator import build
from slate_quantum.operator._diagonal import (
    MomentumOperator,
    MomentumOperatorBasis,
    PositionOperator,
    PositionOperatorBasis,
    Potential,
    RecastBasis,
    RecastDiagonalOperatorBasis,
    momentum_operator_basis,
    position_operator_basis,
    recast_diagonal_basis,
)
from slate_quantum.operator._operator import (
    Operator,
    OperatorList,
    OperatorMetadata,
    operator_basis,
)
from slate_quantum.operator._super_operator import (
    SuperOperator,
    SuperOperatorMetadata,
)
from slate_quantum.operator.build import (
    kinetic_energy_operator as build_kinetic_energy_operator,
)
from slate_quantum.operator.build import (
    kinetic_hamiltonian as build_kinetic_hamiltonian,
)
from slate_quantum.operator.build import (
    repeat_potential,
)
from slate_quantum.operator.linalg import into_diagonal, into_diagonal_hermitian

__all__ = [
    "MomentumOperator",
    "MomentumOperatorBasis",
    "Operator",
    "OperatorList",
    "OperatorMetadata",
    "PositionOperator",
    "PositionOperatorBasis",
    "Potential",
    "RecastBasis",
    "RecastDiagonalOperatorBasis",
    "SuperOperator",
    "SuperOperatorMetadata",
    "build",
    "build_kinetic_energy_operator",
    "build_kinetic_hamiltonian",
    "into_diagonal",
    "into_diagonal_hermitian",
    "momentum_operator_basis",
    "operator_basis",
    "position_operator_basis",
    "recast_diagonal_basis",
    "repeat_potential",
]
