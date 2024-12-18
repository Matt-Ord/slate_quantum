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
from slate_quantum.operator._linalg import (
    commute,
    dagger,
    get_commutator_operator_list,
    into_diagonal,
    into_diagonal_hermitian,
    matmul,
    matmul_list_operator,
    matmul_operator_list,
)
from slate_quantum.operator._operator import (
    Operator,
    OperatorList,
    OperatorMetadata,
    apply,
    apply_to_each,
    expectation,
    expectation_of_each,
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
    "apply",
    "apply_to_each",
    "build",
    "build_kinetic_energy_operator",
    "build_kinetic_hamiltonian",
    "commute",
    "dagger",
    "expectation",
    "expectation_of_each",
    "get_commutator_operator_list",
    "into_diagonal",
    "into_diagonal_hermitian",
    "matmul",
    "matmul_list_operator",
    "matmul_operator_list",
    "momentum_operator_basis",
    "operator_basis",
    "position_operator_basis",
    "recast_diagonal_basis",
    "repeat_potential",
]
