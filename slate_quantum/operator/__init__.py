"""Representation of Quantum Operators."""

from __future__ import annotations

from slate_quantum.operator import _build as build
from slate_quantum.operator import _linalg as linalg
from slate_quantum.operator import _measure as measure
from slate_quantum.operator._build import (
    kinetic_energy as build_kinetic_energy_operator,
)
from slate_quantum.operator._build import (
    kinetic_hamiltonian as build_kinetic_hamiltonian,
)
from slate_quantum.operator._build import (
    repeat_potential,
)
from slate_quantum.operator._diagonal import (
    DiagonalOperatorBasis,
    MomentumOperator,
    MomentumOperatorBasis,
    PositionOperator,
    PositionOperatorBasis,
    Potential,
    RecastBasis,
    momentum_operator_basis,
    position_operator_basis,
    recast_diagonal_basis,
)
from slate_quantum.operator._linalg import (
    commute,
    dagger,
    dagger_each,
    get_commutator_operator_list,
    get_eigenstates_hermitian,
    into_diagonal,
    into_diagonal_hermitian,
    matmul,
    matmul_list_operator,
    matmul_operator_list,
)
from slate_quantum.operator._operator import (
    Operator,
    OperatorBasis,
    OperatorList,
    OperatorListBasis,
    OperatorListMetadata,
    OperatorMetadata,
    SuperOperator,
    SuperOperatorBasis,
    SuperOperatorMetadata,
    apply,
    apply_to_each,
    expectation,
    expectation_of_each,
    operator_basis,
)

__all__ = [
    "DiagonalOperatorBasis",
    "MomentumOperator",
    "MomentumOperatorBasis",
    "Operator",
    "OperatorBasis",
    "OperatorList",
    "OperatorListBasis",
    "OperatorListMetadata",
    "OperatorMetadata",
    "PositionOperator",
    "PositionOperatorBasis",
    "Potential",
    "RecastBasis",
    "SuperOperator",
    "SuperOperatorBasis",
    "SuperOperatorMetadata",
    "apply",
    "apply_to_each",
    "build",
    "build_kinetic_energy_operator",
    "build_kinetic_hamiltonian",
    "commute",
    "dagger",
    "dagger_each",
    "expectation",
    "expectation_of_each",
    "get_commutator_operator_list",
    "get_eigenstates_hermitian",
    "into_diagonal",
    "into_diagonal_hermitian",
    "linalg",
    "matmul",
    "matmul_list_operator",
    "matmul_operator_list",
    "measure",
    "momentum_operator_basis",
    "operator_basis",
    "position_operator_basis",
    "recast_diagonal_basis",
    "repeat_potential",
]
