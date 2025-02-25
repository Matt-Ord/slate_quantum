"""Helpers for building Operators."""

from __future__ import annotations

from slate_quantum.operator._build._hamiltonian import (
    kinetic_energy,
    kinetic_hamiltonian,
)
from slate_quantum.operator._build._momentum import (
    all_filter_scatter,
    filter_scatter,
    k,
    momentum_from_function,
    p,
)
from slate_quantum.operator._build._position import (
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
    x,
    x_displacement_operator,
    x_displacement_operators_stacked,
)
from slate_quantum.operator._build._potential import (
    cos_potential,
    fcc_potential,
    harmonic_potential,
    potential_from_function,
    repeat_potential,
    sin_potential,
    square_potential,
)
from slate_quantum.state._build import get_displacements_x_stacked

__all__ = [
    "all_axis_periodic_operators",
    "all_axis_scattering_operators",
    "all_filter_scatter",
    "all_periodic_operators",
    "all_scattering_operators",
    "axis_periodic_operator",
    "axis_scattering_operator",
    "cos_potential",
    "fcc_potential",
    "filter_scatter",
    "get_displacements_x",
    "get_displacements_x_stacked",
    "harmonic_potential",
    "k",
    "kinetic_energy",
    "kinetic_hamiltonian",
    "momentum_from_function",
    "nx_displacement_operator",
    "nx_displacement_operators_stacked",
    "p",
    "potential_from_function",
    "repeat_potential",
    "scattering_operator",
    "sin_potential",
    "square_potential",
    "total_x_displacement_operator",
    "x",
    "x_displacement_operator",
    "x_displacement_operators_stacked",
]
