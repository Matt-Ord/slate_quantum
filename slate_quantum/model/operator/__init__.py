"""Representation of Quantum Operators."""

from __future__ import annotations

from ._operator import Operator, OperatorList
from .linalg import eig, eigh
from .potential import Potential

__all__ = ["Operator", "OperatorList", "Potential", "eig", "eigh"]
