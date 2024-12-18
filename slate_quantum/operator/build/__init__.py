"""Helpers for building Operators."""

from __future__ import annotations

from slate_quantum.operator.build._hamiltonian import (
    kinetic_energy_operator,
    kinetic_hamiltonian,
)
from slate_quantum.operator.build._momentum import (
    filter_scatter_operator,
    filter_scatter_operators,
    k_operator,
    p_operator,
)
from slate_quantum.operator.build._position import (
    all_axis_periodic_operators,
    all_axis_scattering_operators,
    all_periodic_operators,
    all_scattering_operators,
    axis_periodic_operator,
    axis_scattering_operator,
    get_displacements_x,
    nx_displacement_operator,
    nx_displacement_operators_stacked,
    scattering_operator,
    total_x_displacement_operator,
    x_displacement_operator,
    x_displacement_operators_stacked,
    x_operator,
)
from slate_quantum.operator.build._potential import (
    cos_potential,
    potential_from_function,
    repeat_potential,
    sin_potential,
)
from slate_quantum.state._build import get_displacements_x_stacked

__all__ = [
    "all_axis_periodic_operators",
    "all_axis_scattering_operators",
    "all_periodic_operators",
    "all_scattering_operators",
    "axis_periodic_operator",
    "axis_scattering_operator",
    "cos_potential",
    "filter_scatter_operator",
    "filter_scatter_operators",
    "get_displacements_x",
    "get_displacements_x_stacked",
    "k_operator",
    "kinetic_energy_operator",
    "kinetic_hamiltonian",
    "nx_displacement_operator",
    "nx_displacement_operators_stacked",
    "p_operator",
    "potential_from_function",
    "repeat_potential",
    "scattering_operator",
    "sin_potential",
    "total_x_displacement_operator",
    "x_displacement_operator",
    "x_displacement_operators_stacked",
    "x_operator",
]
