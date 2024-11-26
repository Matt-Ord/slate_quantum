"""
Code for building and manipulating potentials.

A potential is an operator which is diagonal in position basis.
It is represented by an operator in a RecastBasis with
the outer basis being the position basis.
"""

from __future__ import annotations

from ._build import build_cos_potential, repeat_potential
from ._potential import Potential, PotentialBasis

__all__ = [
    "Potential",
    "PotentialBasis",
    "build_cos_potential",
    "repeat_potential",
]
