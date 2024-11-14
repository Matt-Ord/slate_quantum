"""Functions for simulating the dynamics of a quantum system."""

from __future__ import annotations

from .schrodinger import (
    solve_schrodinger_equation,
    solve_schrodinger_equation_decomposition,
)

__all__ = [
    "solve_schrodinger_equation",
    "solve_schrodinger_equation_decomposition",
]
